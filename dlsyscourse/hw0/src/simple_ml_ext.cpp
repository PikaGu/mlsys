#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void matmul(const float* a, const float* b, float* C, size_t m, size_t k, size_t n) {
    for (size_t row = 0; row < m; row++) {
        for (size_t col = 0; col < n; col++) {
            C[row * n + col] = 0;
            for (size_t i = 0; i < k; i++) {
                C[row * n + col] += a[row * k + i] * b[i * n + col];
            }
        }
    }
}

void transpose(const float* in, float* out, size_t m, size_t n) {
    for (size_t row = 0; row < m; row++) {
        for (size_t col = 0; col < n; col++) {
            out[col * m + row] = in[row * n + col];
        }
    }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    float* Z = new float[batch * k];
    float* grad = new float[batch * k];
    float* theta_grad = new float[n * k];
    float* T = new float[n * batch];

    size_t iter_num = m / batch;
    for (size_t iter = 0; iter < iter_num; iter++) {
        const float* iter_X = X + iter * batch * n;
        const unsigned char* iter_y = y + iter * batch;

        // batch*n X n*k
        matmul(iter_X, theta, Z, batch, n, k);
        for (size_t i = 0; i < batch; i++) {
            float sum = 0;
            for (size_t j = 0; j < k; j++) {
                Z[i * k + j] = std::exp(Z[i * k + j]);
                sum += Z[i * k + j];
            }
            for (size_t j = 0; j < k; j++) {
                grad[i * k + j] = Z[i * k + j] / sum;
                if (j == iter_y[i]) {
                    grad[i * k + j] -= 1;
                }
                grad[i * k + j] /= batch;
            }
        }
        transpose(iter_X, T, batch, n);
        // n*batch X batch*k
        matmul(T, grad, theta_grad, n, batch, k);
        for (size_t i = 0; i < n*k; i++) {
            theta[i] -= lr * theta_grad[i];
        }
    }

    delete[] Z;
    delete[] T;
    delete[] grad;
    delete[] theta_grad;
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
