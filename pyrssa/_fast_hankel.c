#include <Python.h>
#include <numpy/arrayobject.h>
#include <fftw3.h>

static PyObject* hankel_mv(PyObject* self, PyObject* args)
{
    PyObject *f_obj, *v_obj;
    if (!PyArg_ParseTuple(args, "OO", &f_obj, &v_obj))
        return NULL;
    PyArrayObject *f_arr = (PyArrayObject*)PyArray_FROM_OTF(f_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *v_arr = (PyArrayObject*)PyArray_FROM_OTF(v_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!f_arr || !v_arr) {
        Py_XDECREF(f_arr);
        Py_XDECREF(v_arr);
        return NULL;
    }
    npy_intp N = PyArray_SIZE(f_arr);
    npy_intp K = PyArray_SIZE(v_arr);
    npy_intp L = N - K + 1;
    if (L <= 0) {
        PyErr_SetString(PyExc_ValueError, "Invalid dimensions");
        Py_DECREF(f_arr);
        Py_DECREF(v_arr);
        return NULL;
    }
    double *F = (double*)PyArray_DATA(f_arr);
    double *v = (double*)PyArray_DATA(v_arr);

    fftw_complex *F_freq = fftw_malloc(sizeof(fftw_complex)*(N/2 + 1));
    fftw_complex *v_freq = fftw_malloc(sizeof(fftw_complex)*(N/2 + 1));
    double *buf = fftw_malloc(sizeof(double)*N);
    if (!F_freq || !v_freq || !buf) {
        fftw_free(F_freq); fftw_free(v_freq); fftw_free(buf);
        PyErr_SetString(PyExc_MemoryError, "Out of memory");
        Py_DECREF(f_arr); Py_DECREF(v_arr);
        return NULL;
    }

    fftw_plan plan_r2c = fftw_plan_dft_r2c_1d(N, buf, F_freq, FFTW_ESTIMATE);
    fftw_plan plan_c2r = fftw_plan_dft_c2r_1d(N, v_freq, buf, FFTW_ESTIMATE);
    if (!plan_r2c || !plan_c2r) {
        fftw_free(F_freq); fftw_free(v_freq); fftw_free(buf);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create FFTW plan");
        Py_DECREF(f_arr); Py_DECREF(v_arr);
        return NULL;
    }

    memcpy(buf, F, sizeof(double)*N);
    fftw_execute_dft_r2c(plan_r2c, buf, F_freq);

    memset(buf, 0, sizeof(double)*N);
    memcpy(buf, v, sizeof(double)*K);
    fftw_execute_dft_r2c(plan_r2c, buf, v_freq);

    for (npy_intp i=0;i<N/2+1;++i) {
        double vr = v_freq[i][0];
        double vi = v_freq[i][1];
        double fr = F_freq[i][0];
        double fi = F_freq[i][1];
        v_freq[i][0] = vr*fr + vi*fi;   /* real part */
        v_freq[i][1] = vr*fi - vi*fr;   /* imag part */
    }

    fftw_execute_dft_c2r(plan_c2r, v_freq, buf);

    npy_intp dims[1] = {L};
    PyObject* out_arr = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    double *out = PyArray_DATA((PyArrayObject*)out_arr);
    for (npy_intp i=0;i<L;++i) out[i] = buf[i]/N;

    fftw_destroy_plan(plan_r2c); fftw_destroy_plan(plan_c2r);
    fftw_free(F_freq); fftw_free(v_freq); fftw_free(buf);
    Py_DECREF(f_arr); Py_DECREF(v_arr);
    return out_arr;
}

static PyMethodDef module_methods[] = {
    {"hankel_mv", hankel_mv, METH_VARARGS, "Multiply Hankel matrix by vector using FFTW"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_fast_hankel",
    NULL,
    -1,
    module_methods
};

PyMODINIT_FUNC PyInit__fast_hankel(void)
{
    import_array();
    return PyModule_Create(&moduledef);
}

