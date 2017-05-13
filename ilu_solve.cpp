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
        std::vector<double>        &x
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
struct sptr_solver {
    int64_t n, nlev;

    std::vector<int64_t> start;
    std::vector<int64_t> order;

    std::vector<int64_t> const &ptr;
    std::vector<int64_t> const &col;
    std::vector<double>  const &val;

    const double *D;

    sptr_solver(
            int64_t n,
            std::vector<int64_t> const &ptr,
            std::vector<int64_t> const &col,
            std::vector<double>  const &val,
            const double *D = 0
            ) :
        n(n), nlev(0), order(n),
        ptr(ptr), col(col), val(val), D(D)
    {
        std::vector<int64_t> lev(n, 0);

        int64_t beg = D ? n-1 : 0;
        int64_t end = D ?  -1 : n;
        int64_t inc = D ?  -1 : 1;

        for(int64_t i = beg; i != end; i += inc) {
            int64_t l = lev[i];

            for(int64_t j = ptr[i]; j < ptr[i+1]; ++j)
                l = std::max(l, lev[col[j]]+1);

            lev[i] = l;
            nlev = std::max(nlev, l+1);
        }

        start.resize(nlev+1, 0);
        for(int64_t i = 0; i < n; ++i) ++start[lev[i]+1];

        std::partial_sum(start.begin(), start.end(), start.begin());

        for(int64_t i = 0; i < n; ++i)
            order[start[lev[i]]++] = i;

        std::rotate(start.begin(), start.end() - 1, start.end());
        start[0] = 0;
    }

    void solve(std::vector<double> &x) const {
        if (D) {
            for(int64_t l = 0; l < nlev; ++l) {
#pragma omp parallel for
                for(int64_t r = start[l]; r < start[l+1]; ++r) {
                    int64_t i = order[r];
                    for(int64_t j = ptr[i], e = ptr[i+1]; j < e; ++j)
                        x[i] -= val[j] * x[col[j]];
                    x[i] = D[i] * x[i];
                }
            }
        } else {
            for(int64_t l = 0; l < nlev; ++l) {
#pragma omp parallel for
                for(int64_t r = start[l]; r < start[l+1]; ++r) {
                    int64_t i = order[r];
                    for(int64_t j = ptr[i], e = ptr[i+1]; j < e; ++j)
                        x[i] -= val[j] * x[col[j]];
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

        void solve(std::vector<double> &x) const {
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

    std::vector<double> xs(n, 1.0);
    std::vector<double> xp(n, 1.0);

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
