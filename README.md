# GRAM: KAN meets Gram Polynomials

GRAM is inspired by Kolmogorov-Arnold Networks ([KAN](https://github.com/KindXiaoming/pykan)) alternatives like [TorchKAN](https://github.com/1ssb/torchkan) and [ChebyKAN](https://github.com/SynodicMonth/ChebyKAN). GRAM introduces a simplified version of the KAN model but leveraging the simplicity of Gram polynomial transformations. What sets it apart from other alternatives is its unique characteristic of being discrete. Unlike other polynomials that are defined on a continuous interval, Gram polynomials stand out as they are defined on a set of discrete points. This discrete nature of GRAM offers a novel way to handle discretized datasets like images, and text data.

### Table of Contents
- [Description](#description)
- [Benchmarks](#benchmarks)
- [Current Issues](#current-issues)
- [What's Next](#whats-next)
- [License](#license)

## Description

## Benchmarks
In this repository, we conduct a series of benchmarks to evaluate different machine learning models on the MNIST dataset. The benchmarks are designed to test each model's accuracy, the number of parameters, and the speed of convergence over a total of 10 epochs.
    In our benchmarking process, we ensure a level playing field by enforcing a uniform architecture across all models. This architecture comprises of the following hidden layers:
    
- [Layer I] (28 * 28, 32)
- [Layer II] (32, 16)
- [Layer III] (16, 10)

## Current Issues

## What's Next

## License

MIT License

Copyright (c) [2024] [Khochawongwat Kongpana]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.