#pragma once
#include <cuda.h>
#include <memory>

template <typename T>
class CUDATensor;

template <typename T>
class Tensor {
   public:
    Tensor(const size_t size, T *data): size_(size), data_(data) {}
    Tensor(const size_t size): Tensor(size, (T*)malloc(sizeof(T) * size)) {}
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor& operator=(Tensor &&other) {
        data_ = std::exchange(other.data_, nullptr);
        size_ = std::exchange(other.size_, 0);
        return *this;
    }
    Tensor(Tensor &&other) {
        *this = std::move(other);
    }

    virtual size_t size() const { return size_; }
    virtual T *data() const { return data_; }

    CUDATensor<T> cuda() {
        CUDATensor<T> tensor(size());
        cudaMemcpy(tensor.data(), data(), sizeof(T) * size(), cudaMemcpyHostToDevice);
        return tensor;
    }
    virtual ~Tensor() noexcept(false) {
        if (data_)
            free(std::exchange(this->data_, nullptr));
    }

   protected:
    Tensor() = default;
    T *data_ = nullptr;
    size_t size_ = 0;

};

template <typename T>
class CUDATensor: public Tensor<T> {
   public:
    CUDATensor(const size_t size, T *value) {
        this->size_ = size;
        this->data_ = value;
    }
    CUDATensor(const size_t size) : CUDATensor(size, nullptr) {
        CUDA_CHECK(cudaMalloc(&this->data_, sizeof(T) * this->size_));
    }

    CUDATensor(const CUDATensor&) = delete;
    CUDATensor& operator=(const CUDATensor&) = delete;
    CUDATensor& operator=(CUDATensor &&other) {
        this->data_ = std::exchange(other.data_, nullptr);
        this->size_ = std::exchange(other.size_, 0);
        return *this;
    }
    CUDATensor(CUDATensor &&other) {
        *this = std::move(other);
    }

    Tensor<T> cpu() {
        Tensor<T> tensor(this->size());
        cudaMemcpy(tensor.data(), this->data(), sizeof(T) * this->size(), cudaMemcpyDeviceToHost);
        return tensor;
    }
    virtual ~CUDATensor() noexcept(false) override {
        if (this->data_)
            CUDA_CHECK(cudaFree(std::exchange(this->data_, nullptr)));
    }
};