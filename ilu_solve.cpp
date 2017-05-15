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
        int64_t beg = D ? n-1 : 0;
        int64_t end = D ?  -1 : n;
        int64_t inc = D ?  -1 : 1;

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
        if (D) {
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
                    x[i] = D[i] * (x[i] - X);
                }
            }
        } else {
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
                    x[i] -= X;
                }
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
struct sptr_solver_v2 {
    struct task {
        int thread_id;
        int64_t row_beg, row_end;
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

    sptr_solver_v2(
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
        int64_t beg = D ? n-1 : 0;
        int64_t end = D ?  -1 : n;
        int64_t inc = D ?  -1 : 1;

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

        // 3.1 Organize matrix rows into tasks.
        //    Each level is split into nthreads tasks.
        prof.tic("schedule");
#ifdef _OPENMP
        int nthreads = omp_get_max_threads();
#else
        int nthreads = 1;
#endif
        std::vector<int64_t> task_id(n);

        {
            scoped_tic tic(prof, "split levels into tasks");
            thread_tasks.resize(nthreads);
            for(int64_t lev = 0; lev < nlev; ++lev) {
                // split each level into tasks.
                int64_t lev_size = start[lev+1] - start[lev];
                int64_t chunk_size = (lev_size + nthreads - 1) / nthreads;

                for(int tid = 0; tid < nthreads; ++tid) {
                    int64_t beg = tid * chunk_size;
                    int64_t end = std::min(beg + chunk_size, lev_size);

                    beg += start[lev];
                    end += start[lev];

                    int64_t this_task = tasks.size();
                    thread_tasks[tid].push_back(this_task);
                    tasks.push_back(task(tid, beg, end));

                    // mark rows that belong to the current task
                    for(int64_t i = beg; i < end; ++i)
                        task_id[order[i]] = this_task;
                }
            }
        }
        prof.toc("schedule");

        // 4. reorganize matrix data for better cache and NUMA locality.
        prof.tic("reorder matrix");
        ptr.resize(n+1, false); ptr[0] = 0;
        col.resize(_ptr[n], false);
        val.resize(_ptr[n], false);

#pragma omp parallel
        {
#ifdef _OPENMP
            int tid = omp_get_thread_num();
#else
            int tid = 0;
#endif
            for(int t : thread_tasks[tid]) {
                int64_t beg = tasks[t].row_beg;
                int64_t end = tasks[t].row_end;

                for(int64_t r = beg; r < end; ++r) {
                    int64_t i = order[r];
                    ptr[r+1] = _ptr[i+1] - _ptr[i];
                }
            }
        }

        std::partial_sum(ptr.data(), ptr.data() + n + 1, ptr.data());

#pragma omp parallel
        {
#ifdef _OPENMP
            int tid = omp_get_thread_num();
#else
            int tid = 0;
#endif
            for(int t : thread_tasks[tid]) {
                int64_t beg = tasks[t].row_beg;
                int64_t end = tasks[t].row_end;

                for(int64_t r = beg; r < end; ++r) {
                    int64_t i = order[r];
                    int64_t h = ptr[r];
                    for(int64_t j = _ptr[i]; j < _ptr[i+1]; ++j) {
                        col[h] = _col[j];
                        val[h] = _val[j];
                        ++h;
                    }
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

            std::vector<int64_t> marker(tasks.size(), -1);

            for(int64_t r = 0; r < n; ++r) {
                int64_t i = order[r];
                int64_t task_i = task_id[i];

                for(int64_t j = ptr[r]; j < ptr[r+1]; ++j) {
                    int64_t task_j = task_id[col[j]];

                    if (marker[task_j] < task_i) {
                        marker[task_j] = task_i;
                        // task_i is parent of task_j:
                        parent[task_i].insert(task_j);
                        child[task_j].insert(task_i);
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

        for(size_t i = 0; i < tasks.size(); ++i) {
            tasks[i].children.assign(child[i].begin(), child[i].end());
            depends[i] = parent[i].size();
        }
        prof.toc("schedule");

    }

    void solve(amgcl::backend::numa_vector<double> &x) const {
#pragma omp parallel for
        for(size_t i = 0; i < tasks.size(); ++i)
            deps[i] = depends[i];

#pragma omp parallel
        {
#ifdef _OPENMP
            int tid = omp_get_thread_num();
#else
            int tid = 0;
#endif
            for(auto t : thread_tasks[tid]) {
                // busy wait for parents to finish:
                while(deps[t] > 0);

                // do the job.
                int64_t beg = tasks[t].row_beg;
                int64_t end = tasks[t].row_end;

                for(int64_t r = beg; r < end; ++r) {
                    int64_t i = order[r];
                    int64_t row_beg = ptr[r];
                    int64_t row_end = ptr[r+1];
                    double X = 0;
                    for(int64_t j = row_beg; j < row_end; ++j)
                        X += val[j] * x[col[j]];
                    if (D)
                        x[i] = D[i] * (x[i] - X);
                    else
                        x[i] -= X;
                }

                // notify children they are free to go
                for(auto c : tasks[t].children)
                    --deps[c];
            }
        }
    }
};

//---------------------------------------------------------------------------
template <class SPTR_Solver>
class ilu_solver {
    private:
        int64_t n;
        std::vector<double>  const &D;
        SPTR_Solver L, U;

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

    std::fill_n(xs.data(), n, 1.0);
    std::fill_n(x1.data(), n, 1.0);
    std::fill_n(x2.data(), n, 1.0);

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

    axpby(1, xs, -1, x1);
    axpby(1, xs, -1, x2);

    std::cout << "delta (v1): " << inner_product(x1, x1) << std::endl;
    std::cout << "delta (v2): " << inner_product(x2, x2) << std::endl;

    std::cout << prof << std::endl;
}
