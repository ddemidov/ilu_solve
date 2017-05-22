#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <numeric>

#include <atomic>

#include <boost/program_options.hpp>

#include <amgcl/io/binary.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/profiler.hpp>

#ifdef _OPENMP
#  include <omp.h>
#endif

typedef amgcl::scoped_tic< amgcl::profiler<> > scoped_tic;
amgcl::profiler<> prof;

//---------------------------------------------------------------------------
// reference implementation
//---------------------------------------------------------------------------
void serial_solve(int64_t n,
        std::vector<int64_t> const &Lptr,
        std::vector<int64_t> const &Lcol,
        std::vector<double>  const &Lval,
        std::vector<int64_t> const &Uptr,
        std::vector<int64_t> const &Ucol,
        std::vector<double>  const &Uval,
        std::vector<double>  const &D,
        amgcl::backend::numa_vector<double> &x
        )
{
    for(int64_t i = 0; i < n; i++) {
        double X = 0;
        for(int64_t j = Lptr[i], e = Lptr[i+1]; j < e; ++j)
            X += Lval[j] * x[Lcol[j]];
        x[i] -= X;
    }

    for(int64_t i = n; i --> 0;) {
        double X = 0;
        for(int64_t j = Uptr[i], e = Uptr[i+1]; j < e; ++j)
            X += Uval[j] * x[Ucol[j]];
        x[i] = D[i] * (x[i] - X);
    }
}

//---------------------------------------------------------------------------
// Solver for sparse triangular systems.
// Uses level scheduling approach.
//---------------------------------------------------------------------------
template <bool lower = true>
struct sptr_solver_v1 {
    int64_t n, nlev;

    std::vector<int64_t> start; // start of each level in order
    amgcl::backend::numa_vector<int64_t> order; // rows ordered by levels

    // matrix diagonal (stored separately). when null, the diagonal is assumed
    // to be filled with ones, and the matrix is assumed to be lower
    // triangular. otherwise matrix is upper triangular.
    const double *D;

    // reordered matrix data:
    amgcl::backend::numa_vector<int64_t> ptr;
    amgcl::backend::numa_vector<int64_t> col;
    amgcl::backend::numa_vector<double>  val;

    sptr_solver_v1(
            int64_t n,
            std::vector<int64_t> const &_ptr,
            std::vector<int64_t> const &_col,
            std::vector<double>  const &_val,
            const double *D = 0
            ) :
        n(n), nlev(0), order(n, false), D(D)
    {
        std::vector<int64_t> lev(n, 0);

        // 1. split rows into levels.
        prof.tic("color");
        int64_t beg = lower ? 0 : n-1;
        int64_t end = lower ? n :  -1;
        int64_t inc = lower ? 1 :  -1;

        for(int64_t i = beg; i != end; i += inc) {
            int64_t l = lev[i];

            for(int64_t j = _ptr[i]; j < _ptr[i+1]; ++j)
                l = std::max(l, lev[_col[j]]+1);

            lev[i] = l;
            nlev = std::max(nlev, l+1);
        }
        prof.toc("color");

        // 2. reorder matrix rows.
        prof.tic("sort");
        start.resize(nlev+1, 0);
        for(int64_t i = 0; i < n; ++i) ++start[lev[i]+1];

        // numa-touch order vector.
        for(int64_t l = 0; l < nlev; ++l) {
            int64_t lev_beg = start[l];
            int64_t lev_end = start[l+1] = lev_beg + start[l+1];
#pragma omp parallel for
            for(int64_t r = lev_beg; r < lev_end; ++r) {
                order[r] = 0;
            }
        }

        for(int64_t i = 0; i < n; ++i)
            order[start[lev[i]]++] = i;

        std::rotate(start.begin(), start.end() - 1, start.end());
        start[0] = 0;
        prof.toc("sort");

        // 3. reorganize matrix data for better cache and NUMA locality.
        prof.tic("reorder matrix");
        ptr.resize(n+1, false); ptr[0] = 0;
        col.resize(_ptr[n], false);
        val.resize(_ptr[n], false);

        for(int64_t l = 0; l < nlev; ++l) {
            int64_t lev_beg = start[l];
            int64_t lev_end = start[l+1];
#pragma omp parallel for
            for(int64_t r = lev_beg; r < lev_end; ++r) {
                int64_t i = order[r];
                ptr[r+1] = _ptr[i+1] - _ptr[i];
            }
        }

        std::partial_sum(ptr.data(), ptr.data() + n + 1, ptr.data());

        for(int64_t l = 0; l < nlev; ++l) {
            int64_t lev_beg = start[l];
            int64_t lev_end = start[l+1];
#pragma omp parallel for
            for(int64_t r = lev_beg; r < lev_end; ++r) {
                int64_t i = order[r];
                int64_t h = ptr[r];
                for(int64_t j = _ptr[i]; j < _ptr[i+1]; ++j) {
                    col[h] = _col[j];
                    val[h] = _val[j];
                    ++h;
                }
            }
        }
        prof.toc("reorder matrix");
    }

    void solve(amgcl::backend::numa_vector<double> &x) const {
        for(int64_t l = 0; l < nlev; ++l) {
            int64_t lev_beg = start[l];
            int64_t lev_end = start[l+1];
#pragma omp parallel for
            for(int64_t r = lev_beg; r < lev_end; ++r) {
                int64_t i = order[r];
                int64_t row_beg = ptr[r];
                int64_t row_end = ptr[r+1];
                double X = 0;
                for(int64_t j = row_beg; j < row_end; ++j)
                    X += val[j] * x[col[j]];

                if (lower)
                    x[i] -= X;
                else
                    x[i] = D[i] * (x[i] - X);
            }
        }
    }
};

//---------------------------------------------------------------------------
// Solver for sparse triangular systems.
// Uses task scheduling approach from [1].
//
// [1] Park, Jongsoo, et al. "Sparsifying synchronization for high-performance
//     shared-memory sparse triangular solver." International Supercomputing
//     Conference. Springer International Publishing, 2014.
//---------------------------------------------------------------------------
template <bool lower>
struct sptr_solver_v2 {
    struct task {
        int thread_id;
        int64_t row_beg, row_end, loc_beg;
        std::vector<int> children;

        task(int id, int64_t beg, int64_t end)
            : thread_id(id), row_beg(beg), row_end(end)
        {}
    };

    std::vector<task> tasks;
    std::vector< std::vector<int> > thread_tasks;
    std::vector<int> depends;
    mutable std::vector< std::atomic<int> > deps;

    int64_t n, nlev;

    amgcl::backend::numa_vector<int64_t> order; // rows ordered by levels

    // reordered matrix data:
    std::vector< std::vector<int64_t> > ptr;
    std::vector< std::vector<int64_t> > col;
    std::vector< std::vector<double>  > val;
    std::vector< std::vector<double>  > D;

    sptr_solver_v2(
            int64_t n,
            std::vector<int64_t> const &_ptr,
            std::vector<int64_t> const &_col,
            std::vector<double>  const &_val,
            const double *_D = 0
            ) :
        n(n), nlev(0), order(n, false)
    {
        std::vector<int64_t> lev(n, 0);

        // 1. split rows into levels.
        prof.tic("color");
        int64_t beg = lower ? 0 : n-1;
        int64_t end = lower ? n :  -1;
        int64_t inc = lower ? 1 :  -1;

        for(int64_t i = beg; i != end; i += inc) {
            int64_t l = lev[i];

            for(int64_t j = _ptr[i]; j < _ptr[i+1]; ++j)
                l = std::max(l, lev[_col[j]]+1);

            lev[i] = l;
            nlev = std::max(nlev, l+1);
        }
        prof.toc("color");

        // 2. reorder matrix rows.
        prof.tic("sort");
        std::vector<int64_t> start(nlev+1, 0); // start of each level in ordered rows
        for(int64_t i = 0; i < n; ++i) ++start[lev[i]+1];

        // numa-touch order vector.
        for(int64_t l = 0; l < nlev; ++l) {
            int64_t lev_beg = start[l];
            int64_t lev_end = start[l+1] = lev_beg + start[l+1];
#pragma omp parallel for
            for(int64_t r = lev_beg; r < lev_end; ++r) {
                order[r] = 0;
            }
        }

        for(int64_t i = 0; i < n; ++i)
            order[start[lev[i]]++] = i;

        std::rotate(start.begin(), start.end() - 1, start.end());
        start[0] = 0;
        prof.toc("sort");

        // 3.1 Organize matrix rows into tasks.
        //    Each level is split into nthreads tasks.
        prof.tic("schedule");
        int nthreads = num_threads();
        std::vector<int64_t> thread_rows(nthreads, 0);
        std::vector<int64_t> thread_cols(nthreads, 0);
        std::vector<int64_t> task_id(n);

        {
            scoped_tic tic(prof, "split levels into tasks");
            thread_tasks.resize(nthreads);
#pragma omp parallel
            {
                int tid = thread_id();
                thread_tasks[tid].resize(nlev);
                thread_tasks[tid].resize(0);
            }

            for(int64_t lev = 0; lev < nlev; ++lev) {
                // split each level into tasks.
                int64_t lev_size = start[lev+1] - start[lev];
                int64_t chunk_size = (lev_size + nthreads - 1) / nthreads;

                for(int tid = 0; tid < nthreads; ++tid) {
                    int64_t beg = std::min(tid * chunk_size, lev_size);
                    int64_t end = std::min(beg + chunk_size, lev_size);

                    beg += start[lev];
                    end += start[lev];

                    thread_rows[tid] += end - beg;

                    int64_t this_task = tasks.size();
                    thread_tasks[tid].push_back(this_task);
                    tasks.push_back(task(tid, beg, end));

                    // mark rows that belong to the current task
                    for(int64_t i = beg; i < end; ++i) {
                        int64_t j = order[i];
                        task_id[j] = this_task;
                        thread_cols[tid] += _ptr[j+1] - _ptr[j];
                    }
                }
            }
        }
        prof.toc("schedule");

        // 4. reorganize matrix data for better cache and NUMA locality.
        prof.tic("reorder matrix");
        ptr.resize(nthreads);
        col.resize(nthreads);
        val.resize(nthreads);
        if (!lower) D.resize(nthreads);

#pragma omp parallel
        {
            int tid = thread_id();
            ptr[tid].reserve(thread_rows[tid] + 1);
            col[tid].reserve(thread_cols[tid]);
            val[tid].reserve(thread_cols[tid]);
            ptr[tid].push_back(0);

            if (!lower) D[tid].reserve(thread_rows[tid]);

            for(int t : thread_tasks[tid]) {
                int64_t beg = tasks[t].row_beg;
                int64_t end = tasks[t].row_end;

                tasks[t].loc_beg = ptr[tid].size() - 1;

                for(int64_t r = beg; r < end; ++r) {
                    int64_t i = order[r];
                    if (!lower) D[tid].push_back(_D[i]);
                    for(int64_t j = _ptr[i]; j < _ptr[i+1]; ++j) {
                        col[tid].push_back(_col[j]);
                        val[tid].push_back(_val[j]);
                    }
                    ptr[tid].push_back(col[tid].size());
                }
            }
        }
        prof.toc("reorder matrix");

        // 3.2. Build task dependency graphs (both directions).
        prof.tic("schedule");
        std::vector< std::set<int64_t> > parent(tasks.size());
        std::vector< std::set<int64_t> > child(tasks.size());
        {
            scoped_tic tic(prof, "build TDG");
            depends.resize(tasks.size());
            std::vector<std::atomic<int>>(tasks.size()).swap(deps);

            std::vector<size_t> marker(tasks.size(), -1);

            for(size_t i = 0; i < tasks.size(); ++i) {
                int64_t t_id = tasks[i].thread_id;
                int64_t gbeg = tasks[i].row_beg;
                int64_t gend = tasks[i].row_end;
                int64_t lbeg = tasks[i].loc_beg;

                for(int64_t r = gbeg, k = lbeg; r < gend; ++r, ++k) {
                    for(int64_t j = ptr[t_id][k]; j < ptr[t_id][k+1]; ++j) {
                        int64_t task_j = task_id[col[t_id][j]];

                        if (marker[task_j] != i) {
                            marker[task_j] = i;
                            // task_i is parent of task_j:
                            parent[i].insert(task_j);
                            child[task_j].insert(i);
                        }
                    }
                }
            }
        }

        {
            scoped_tic tic(prof, "sparsify TDG");
            for(size_t i = 0; i < tasks.size(); ++i) {
                for(auto k : child[i]) {
                    for(auto j : parent[k]) {
                        if (child[i].count(j)) {
                            child[i].erase(k);
                            parent[k].erase(i);
                            break;
                        }
                    }
                }
            }

            for(size_t i = 0; i < tasks.size(); ++i) {
                for(auto j : child[i]) {
                    if (tasks[i].thread_id == tasks[j].thread_id) {
                        child[i].erase(j);
                        parent[j].erase(i);
                    }
                }
            }
        }

#pragma omp parallel
        {
            int tid = thread_id();
            for(int t : thread_tasks[tid]) {
                tasks[t].children.assign(child[t].begin(), child[t].end());
                depends[t] = parent[t].size();
            }
        }
        prof.toc("schedule");

    }

    void solve(amgcl::backend::numa_vector<double> &x) const {
#pragma omp parallel for
        for(size_t i = 0; i < tasks.size(); ++i)
            deps[i] = depends[i];

#pragma omp parallel
        {
            int tid = thread_id();
            for(auto t : thread_tasks[tid]) {
                // busy wait for parents to finish:
                while(deps[t] > 0);

                // do the job.
                int64_t beg = tasks[t].row_beg;
                int64_t end = tasks[t].row_end;

                for(int64_t r = beg, k = tasks[t].loc_beg; r < end; ++r, ++k) {
                    int64_t i = order[r];
                    int64_t row_beg = ptr[tid][k];
                    int64_t row_end = ptr[tid][k+1];
                    double X = 0;
                    for(int64_t j = row_beg; j < row_end; ++j) {
                        X += val[tid][j] * x[col[tid][j]];
                    }
                    if (lower)
                        x[i] -= X;
                    else
                        x[i] = D[tid][k] * (x[i] - X);
                }

                // notify children they are free to go
                for(auto c : tasks[t].children)
                    --deps[c];
            }
        }
    }

    static int num_threads() {
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }

    static int thread_id() {
#ifdef _OPENMP
        return omp_get_thread_num();
#else
        return 0;
#endif
    }
};

template <bool lower>
struct sptr_solver_v3 {
    struct task {
        int64_t beg, end;
        task(int64_t beg, int64_t end) : beg(beg), end(end) {}
    };

    int nthreads;

    // thread-specific storage:
    std::vector< std::vector<task>    > tasks;
    std::vector< std::vector<int64_t> > ptr;
    std::vector< std::vector<int64_t> > col;
    std::vector< std::vector<double>  > val;
    std::vector< std::vector<int64_t> > ord;
    std::vector< std::vector<double>  > D;

    sptr_solver_v3(
        int64_t n,
        std::vector<int64_t> const &_ptr,
        std::vector<int64_t> const &_col,
        std::vector<double>  const &_val,
        const double *_D = 0)
      : nthreads(num_threads()), tasks(nthreads),
        ptr(nthreads), col(nthreads), val(nthreads), ord(nthreads)
    {
        int64_t nlev = 0;

        std::vector<int64_t> level(n, 0);
        std::vector<int64_t> order(n, 0);


        // 1. split rows into levels.
        int64_t beg = lower ? 0 : n-1;
        int64_t end = lower ? n :  -1;
        int64_t inc = lower ? 1 :  -1;

        for(int64_t i = beg; i != end; i += inc) {
            int64_t l = level[i];

            for(int64_t j = _ptr[i]; j < _ptr[i+1]; ++j)
                l = std::max(l, level[_col[j]]+1);

            level[i] = l;
            nlev = std::max(nlev, l+1);
        }


        // 2. reorder matrix rows.
        std::vector<int64_t> start(nlev+1, 0);

        for(int64_t i = 0; i < n; ++i)
            ++start[level[i]+1];

        std::partial_sum(start.begin(), start.end(), start.begin());

        for(int64_t i = 0; i < n; ++i)
            order[start[level[i]]++] = i;

        std::rotate(start.begin(), start.end() - 1, start.end());
        start[0] = 0;


        // 3. Organize matrix rows into tasks.
        //    Each level is split into nthreads tasks.
        std::vector<int64_t> thread_rows(nthreads, 0);
        std::vector<int64_t> thread_cols(nthreads, 0);

#pragma omp parallel
        {
            int tid = thread_id();
            tasks[tid].reserve(nlev);

            for(int64_t lev = 0; lev < nlev; ++lev) {
                // split each level into tasks.
                int64_t lev_size = start[lev+1] - start[lev];
                int64_t chunk_size = (lev_size + nthreads - 1) / nthreads;

                int64_t beg = std::min(tid * chunk_size, lev_size);
                int64_t end = std::min(beg + chunk_size, lev_size);

                beg += start[lev];
                end += start[lev];

                tasks[tid].push_back(task(beg, end));

                // count rows and nonzeros in the current task
                thread_rows[tid] += end - beg;
                for(int64_t i = beg; i < end; ++i) {
                    int64_t j = order[i];
                    thread_cols[tid] += _ptr[j+1] - _ptr[j];
                }
            }
        }

        // 4. reorganize matrix data for better cache and NUMA locality.
        if (!lower) D.resize(nthreads);

#pragma omp parallel
        {
            int tid = thread_id();

            col[tid].reserve(thread_cols[tid]);
            val[tid].reserve(thread_cols[tid]);
            ord[tid].reserve(thread_rows[tid]);
            ptr[tid].reserve(thread_rows[tid] + 1);
            ptr[tid].push_back(0);

            if (!lower) D[tid].reserve(thread_rows[tid]);

            for(auto &t : tasks[tid]) {
                int64_t loc_beg = ptr[tid].size() - 1;
                int64_t loc_end = loc_beg;

                for(int64_t r = t.beg; r < t.end; ++r, ++loc_end) {
                    int64_t i = order[r];
                    if (!lower) D[tid].push_back(_D[i]);

                    ord[tid].push_back(i);

                    for(int64_t j = _ptr[i]; j < _ptr[i+1]; ++j) {
                        col[tid].push_back(_col[j]);
                        val[tid].push_back(_val[j]);
                    }

                    ptr[tid].push_back(col[tid].size());
                }

                t.beg = loc_beg;
                t.end = loc_end;
            }
        }
    }

    template <class Vector>
    void solve(Vector &x) const {
#pragma omp parallel
        {
            int tid = thread_id();

            for(const auto &t : tasks[tid]) {
                for(int64_t r = t.beg; r < t.end; ++r) {
                    int64_t i   = ord[tid][r];
                    int64_t beg = ptr[tid][r];
                    int64_t end = ptr[tid][r+1];

                    double X = 0.0;
                    for(int64_t j = beg; j < end; ++j)
                        X += val[tid][j] * x[col[tid][j]];

                    if (lower)
                        x[i] -= X;
                    else
                        x[i] = D[tid][r] * (x[i] - X);
                }

                // each task corresponds to a level, so we need
                // to synchronize across threads at this point:
#pragma omp barrier
            }
        }
    }

    static int num_threads() {
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }

    static int thread_id() {
#ifdef _OPENMP
        return omp_get_thread_num();
#else
        return 0;
#endif
    }
};

//---------------------------------------------------------------------------
template <template <bool> class SPTR_Solver>
class ilu_solver {
    private:
        int64_t n;
        std::vector<double>  const &D;
        SPTR_Solver<true>  L;
        SPTR_Solver<false> U;

    public:
        ilu_solver(
                int64_t n,
                std::vector<int64_t> const &Lptr,
                std::vector<int64_t> const &Lcol,
                std::vector<double>  const &Lval,
                std::vector<int64_t> const &Uptr,
                std::vector<int64_t> const &Ucol,
                std::vector<double>  const &Uval,
                std::vector<double>  const &D
                ) :
            n(n), D(D),
            L(n, Lptr, Lcol, Lval),
            U(n, Uptr, Ucol, Uval, D.data())
        {
        }

        void solve(amgcl::backend::numa_vector<double> &x) const {
            L.solve(x);
            U.solve(x);
        }
};

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    namespace po = boost::program_options;
    namespace io = amgcl::io;

    po::options_description desc("Options");

    desc.add_options()
        ("help,h", "Show this help")
        (
         "iters,n",
         po::value<int>()->default_value(10),
         "Number of solves to measure"
        )
        (
         "D",
         po::value<std::string>()->default_value("ilu_d.bin"),
         "Diagonal of the upper triangular matrix U"
        )
        (
         "L",
         po::value<std::string>()->default_value("ilu_l.bin"),
         "The lower triangular matrix L"
        )
        (
         "U",
         po::value<std::string>()->default_value("ilu_u.bin"),
         "The upper triangular matrix U"
        )
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    std::vector<double> D;

    std::vector<int64_t> Lptr;
    std::vector<int64_t> Lcol;
    std::vector<double>  Lval;

    std::vector<int64_t> Uptr;
    std::vector<int64_t> Ucol;
    std::vector<double>  Uval;

    int64_t n, m;

    {
        scoped_tic t(prof, "read");
        io::read_dense(vm["D"].as<std::string>(), n, m, D);
        io::read_crs  (vm["L"].as<std::string>(), n, Lptr, Lcol, Lval);
        io::read_crs  (vm["U"].as<std::string>(), n, Uptr, Ucol, Uval);
    }

    amgcl::backend::numa_vector<double> xs(n, true);
    amgcl::backend::numa_vector<double> x1(n, true);
    amgcl::backend::numa_vector<double> x2(n, true);
    amgcl::backend::numa_vector<double> x3(n, true);

    std::fill_n(xs.data(), n, 1.0);
    std::fill_n(x1.data(), n, 1.0);
    std::fill_n(x2.data(), n, 1.0);
    std::fill_n(x3.data(), n, 1.0);

    const int niters = vm["iters"].as<int>();

    {
        std::cout << "serial..." << std::endl;
        scoped_tic t(prof, "serial solve");
        for(int i = 0; i < niters; ++i)
            serial_solve(n, Lptr, Lcol, Lval, Uptr, Ucol, Uval, D, xs);
    }

    {
        std::cout << "parallel (level scheduling)..." << std::endl;
        scoped_tic t1(prof, "parallel (level scheduling)");

        prof.tic("setup");
        ilu_solver<sptr_solver_v1> S(n, Lptr, Lcol, Lval, Uptr, Ucol, Uval, D);
        prof.toc("setup");

        scoped_tic t2(prof, "solve");
        for(int i = 0; i < niters; ++i)
            S.solve(x1);
    }

    {
        std::cout << "parallel (task scheduling)..." << std::endl;
        scoped_tic t1(prof, "parallel (task scheduling)");

        prof.tic("setup");
        ilu_solver<sptr_solver_v2> S(n, Lptr, Lcol, Lval, Uptr, Ucol, Uval, D);
        prof.toc("setup");

        scoped_tic t2(prof, "solve");
        for(int i = 0; i < niters; ++i)
            S.solve(x2);
    }

    {
        std::cout << "parallel (v3)..." << std::endl;
        scoped_tic t1(prof, "parallel (v3)");

        prof.tic("setup");
        ilu_solver<sptr_solver_v3> S(n, Lptr, Lcol, Lval, Uptr, Ucol, Uval, D);
        prof.toc("setup");

        scoped_tic t2(prof, "solve");
        for(int i = 0; i < niters; ++i)
            S.solve(x3);
    }

    axpby(1, xs, -1, x1);
    axpby(1, xs, -1, x2);
    axpby(1, xs, -1, x3);

    std::cout << "delta (v1): " << inner_product(x1, x1) << std::endl;
    std::cout << "delta (v2): " << inner_product(x2, x2) << std::endl;
    std::cout << "delta (v3): " << inner_product(x3, x3) << std::endl;

    std::cout << prof << std::endl;
}
