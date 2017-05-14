#include <iostream>
#include <boost/program_options.hpp>
#include <amgcl/io/binary.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/profiler.hpp>

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

#ifdef REORDER_MATRICES
    // reordered matrix data:
    amgcl::backend::numa_vector<int64_t> ptr;
    amgcl::backend::numa_vector<int64_t> col;
    amgcl::backend::numa_vector<double>  val;
#else
    std::vector<int64_t> const &ptr;
    std::vector<int64_t> const &col;
    std::vector<double>  const &val;
#endif

    sptr_solver_v1(
            int64_t n,
            std::vector<int64_t> const &_ptr,
            std::vector<int64_t> const &_col,
            std::vector<double>  const &_val,
            const double *D = 0
            ) :
        n(n), nlev(0), order(n, false), D(D)
#ifndef REORDER_MATRICES
        , ptr(_ptr), col(_col), val(_val)
#endif
    {
        std::vector<int64_t> lev(n, 0);

        // 1. split rows into levels.
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

        // 2. reorder matrix rows.
        start.resize(nlev+1, 0);
        for(int64_t i = 0; i < n; ++i) ++start[lev[i]+1];

        std::partial_sum(start.begin(), start.end(), start.begin());

        // numa-touch order vector.
        for(int64_t l = 0; l < nlev; ++l) {
            int64_t lev_beg = start[l];
            int64_t lev_end = start[l+1];
#pragma omp parallel for
            for(int64_t r = lev_beg; r < lev_end; ++r) {
                order[r] = 0;
            }
        }

        for(int64_t i = 0; i < n; ++i)
            order[start[lev[i]]++] = i;

        std::rotate(start.begin(), start.end() - 1, start.end());
        start[0] = 0;

#ifdef REORDER_MATRICES
        // 3. reorganize matrix data for better cache and NUMA locality.
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
#endif
    }

    void solve(amgcl::backend::numa_vector<double> &x) const {
        if (D) {
            for(int64_t l = 0; l < nlev; ++l) {
                int64_t lev_beg = start[l];
                int64_t lev_end = start[l+1];
#pragma omp parallel for
                for(int64_t r = lev_beg; r < lev_end; ++r) {
                    int64_t i = order[r];
#ifdef REORDER_MATRICES
                    int64_t row_beg = ptr[r];
                    int64_t row_end = ptr[r+1];
#else
                    int64_t row_beg = ptr[i];
                    int64_t row_end = ptr[i+1];
#endif
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
#ifdef REORDER_MATRICES
                    int64_t row_beg = ptr[r];
                    int64_t row_end = ptr[r+1];
#else
                    int64_t row_beg = ptr[i];
                    int64_t row_end = ptr[i+1];
#endif
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

    std::fill_n(xs.data(), n, 1.0);
    std::fill_n(x1.data(), n, 1.0);

    const int niters = vm["iters"].as<int>();

    {
        scoped_tic t(prof, "serial solve");
        for(int i = 0; i < niters; ++i)
            serial_solve(n, Lptr, Lcol, Lval, Uptr, Ucol, Uval, D, xs);
    }

    {
        scoped_tic t1(prof, "parallel (level scheduling)");

        prof.tic("setup");
        ilu_solver<sptr_solver_v1> S(n, Lptr, Lcol, Lval, Uptr, Ucol, Uval, D);
        prof.toc("setup");

        scoped_tic t2(prof, "solve");
        for(int i = 0; i < niters; ++i)
            S.solve(x1);
    }

    axpby(1, xs, -1, x1);
    std::cout << "delta (v1): " << inner_product(x1, x1) << std::endl;

    std::cout << prof << std::endl;
}
