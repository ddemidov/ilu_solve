#include <iostream>
#include <amgcl/io/binary.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/profiler.hpp>

typedef amgcl::scoped_tic< amgcl::profiler<> > scoped_tic;
amgcl::profiler<> prof;

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
        for(int64_t j = Lptr[i], e = Lptr[i+1]; j < e; ++j)
            x[i] -= Lval[j] * x[Lcol[j]];
    }

    for(int64_t i = n; i --> 0;) {
        for(int64_t j = Uptr[i], e = Uptr[i+1]; j < e; ++j)
            x[i] -= Uval[j] * x[Ucol[j]];
        x[i] = D[i] * x[i];
    }
}

//---------------------------------------------------------------------------
// Solver for sparse triangular systems.
// Uses level scheduling approach.
//---------------------------------------------------------------------------
struct sptr_solver {
    int64_t n, nlev;

    std::vector<int64_t> start; // start of each level in order
    amgcl::backend::numa_vector<int64_t> order; // rows ordered by levels

    // reordered matrix data:
    amgcl::backend::numa_vector<int64_t> ptr;
    amgcl::backend::numa_vector<int64_t> col;
    amgcl::backend::numa_vector<double>  val;

    // matrix diagonal (stored separately). when null, the diagonal is assumed
    // to be filled with ones, and the matrix is assumed to be lower
    // triangular. otherwise matrix is upper triangular.
    const double *D;

    sptr_solver(
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
    }

    void solve(amgcl::backend::numa_vector<double> &x) const {
        if (D) {
            for(int64_t l = 0; l < nlev; ++l) {
                int64_t lev_beg = start[l];
                int64_t lev_end = start[l+1];
#pragma omp parallel for
                for(int64_t r = lev_beg; r < lev_end; ++r) {
                    int64_t i = order[r];
                    double X = 0;
                    for(int64_t j = ptr[r], e = ptr[r+1]; j < e; ++j)
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
                    double X = 0;
                    for(int64_t j = ptr[r], e = ptr[r+1]; j < e; ++j)
                        X += val[j] * x[col[j]];
                    x[i] -= X;
                }
            }
        }
    }
};

//---------------------------------------------------------------------------
class ilu_solver {
    private:
        int64_t n;
        std::vector<double>  const &D;
        sptr_solver L, U;

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
    const char *dfile = argc < 2 ? "ilu_d.bin" : argv[1];
    const char *lfile = argc < 3 ? "ilu_l.bin" : argv[2];
    const char *ufile = argc < 4 ? "ilu_u.bin" : argv[3];

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
        amgcl::io::read_dense(dfile, n, m, D);
        amgcl::io::read_crs(lfile, n, Lptr, Lcol, Lval);
        amgcl::io::read_crs(ufile, n, Uptr, Ucol, Uval);
    }

    amgcl::backend::numa_vector<double> xs(n, true);
    amgcl::backend::numa_vector<double> xp(n, true);

    std::fill_n(xs.data(), n, 1.0);
    std::fill_n(xp.data(), n, 1.0);

    {
        scoped_tic t(prof, "serial");
        serial_solve(n, Lptr, Lcol, Lval, Uptr, Ucol, Uval, D, xs);
    }

    {
        scoped_tic t1(prof, "parallel");
        ilu_solver S(n, Lptr, Lcol, Lval, Uptr, Ucol, Uval, D);

        scoped_tic t2(prof, "solve");
        S.solve(xp);
    }

    double delta = 0;
    for(int64_t i = 0; i < n; ++i)
        delta += (xs[i] - xp[i]) * (xs[i] - xp[i]);

    std::cout << "delta: " << delta << std::endl;

    std::cout << prof << std::endl;
}
