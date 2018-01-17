#if 0
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define SYNC_ON_DEV 0
#else
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#define SYNC_ON_DEV 0
#endif
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/cl2.hpp>
#if CL_HPP_TARGET_OPENCL_VERSION >= 200
namespace cl {
    namespace detail {
        CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE, size_t)
        CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE, size_t)
    }
}
#endif

#include <cstdio>
#include <cstdlib>

#include <string>
#include <array>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <streambuf>

#if CL_HPP_TARGET_OPENCL_VERSION >= 200
static const size_t PLATFORM_INDEX = 1;
#else
static const size_t PLATFORM_INDEX = 0;
#endif

static const size_t W = 1024;
static const size_t H = 1024;

#if SYNC_ON_DEV
const cl::NDRange kernelRangeGlobal(W, H);
#else
const cl::NDRange kernelRangeGlobal(16, H);
#endif
const cl::NDRange kernelRangeLocal(16, 16);

#define kernelFile "src/pred.cl"
#define kernelName "pred"

static const std::string DUMP_FILE     {"dump"};
static const std::string DUMP_FILE1000 {"dump_after1000"};
static const std::string DUMP_FILE_EXT {".yuv"};


int main(void)
{
    cl_int err = CL_SUCCESS;
    try {
        // Platforms & Context
        // --------------------------------------------
        std::cout << "=======================================" << std::endl;
        std::cout << "Platforms" << std::endl;
        std::cout << "=======================================" << std::endl;
        std::vector<cl::Platform> platforms;
        err |= cl::Platform::get(&platforms);
        for (auto &plat : platforms) {
            std::array<std::string, 5> params {{
                plat.getInfo<CL_PLATFORM_PROFILE>(),
                plat.getInfo<CL_PLATFORM_VERSION>(),
                plat.getInfo<CL_PLATFORM_NAME>(),
                plat.getInfo<CL_PLATFORM_VENDOR>(),
                plat.getInfo<CL_PLATFORM_EXTENSIONS>(),
            }};
            for (auto &param : params) {
                std::cout << param << std::endl;
            }

            std::cout << "--------------------" << std::endl;
        }
        if (platforms.size() == 0 || PLATFORM_INDEX >= platforms.size()) {
            std::cout << "ERROR: No platforms" << std::endl;
            return EXIT_FAILURE;
        }
        std::cout << std::endl;

        cl::Platform plat = cl::Platform::setDefault(platforms[PLATFORM_INDEX]);
        if (plat != platforms[PLATFORM_INDEX]) {
            std::cerr << "ERROR: Setting default platform: " << PLATFORM_INDEX << std::endl;
            return EXIT_FAILURE;
        }
        std::cout << "Use platform " << PLATFORM_INDEX << std::endl;
        std::cout << std::endl;
        // Context
        cl::Context context = cl::Context::getDefault();

        // Devices
        // --------------------------------------------
        std::cout << "=======================================" << std::endl;
        std::cout << "Devices" << std::endl;
        std::cout << "=======================================" << std::endl;
        auto devices = context.getInfo<CL_CONTEXT_DEVICES>();
        size_t i = 0;
        for (auto &device : devices) {
            std::cout << "--------------------" << std::endl;
            std::cout << i << std::endl;
            std::cout << "--------------------" << std::endl;

            std::array<std::string, 11> params {{
                device.getInfo<CL_DEVICE_NAME>(),
                device.getInfo<CL_DEVICE_VENDOR>(),
                device.getInfo<CL_DEVICE_PROFILE>(),
                device.getInfo<CL_DEVICE_VERSION>(),
                device.getInfo<CL_DRIVER_VERSION>(),
                device.getInfo<CL_DEVICE_OPENCL_C_VERSION>(),
                std::to_string(device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()) + "[Cores] @ " + std::to_string(device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>()) + "[MHz]",
#if CL_HPP_TARGET_OPENCL_VERSION < 200
                "Host Unified Memory: " + std::to_string(device.getInfo<CL_DEVICE_HOST_UNIFIED_MEMORY>()),
                "CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE: " + std::string{ "?" },
                "CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE: " + std::string{ "?" },
#else
                "Host Unified Memory: " + std::string{ "?" },
                "CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE: " + std::to_string(device.getInfo<CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE>()),
                "CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE: " + std::to_string(device.getInfo<CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE>()),
#endif
                device.getInfo<CL_DEVICE_EXTENSIONS>(),
            }};
            for (auto &param : params) {
                std::cout << param << std::endl;
            }
            ++i;
        }
        std::cout << std::endl;

        // Build
        // --------------------------------------------
        std::ifstream from(kernelFile);
        std::string kernelStr((std::istreambuf_iterator<char>(from)),
                               std::istreambuf_iterator<char>());
        from.close();
        cl::Program::Sources sources {kernelStr};
        cl::Program program = cl::Program(sources);
        try {
            err |= program.build("");
        } catch (cl::Error err) {
            std::cerr
            << "ERROR: "
            << err.what()
            << "("
            << err.err()
            << ")"
            << std::endl;

            cl_int buildErr = CL_SUCCESS;
            auto buildInfo = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
            for (auto &pair : buildInfo) {
                std::cerr << pair.second << std::endl;
            }
            return EXIT_FAILURE;
        }

        // Execution
        // --------------------------------------------
        cl::CommandQueue queue(
            cl::QueueProperties::Profiling //cl::QueueProperties::None
        );
        cl::CommandQueue q = cl::CommandQueue::setDefault(queue);
        if (q != queue) {
            std::cerr << "ERROR: Setting default queue" << std::endl;
            return EXIT_FAILURE;
        }

        auto kernelFunc = cl::KernelFunctor<cl::Buffer, int>(program, kernelName);
        auto kernel = kernelFunc.getKernel();
        size_t s = kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(cl::Device::getDefault());
        std::cout << "CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: " << s << std::endl;

        cl::Buffer yPlaneDev(CL_MEM_READ_WRITE, W * H);
#if SYNC_ON_DEV
        // Sync on Dev
        kernelFunc(
            cl::EnqueueArgs(
                kernelRangeGlobal,
                kernelRangeLocal
            ),
            yPlaneDev,
            0,
            err
        );
#else
        // Sync on Host
        // -> xth
        // 0(0<<1): 0 1 2 3 4 5 6 7 8 9
        // 1(1<<1):     2 3 4
        // 2(2<<1):         4 5 6
        size_t nx = W / 16 + ((H / 16 - 1) * 2);
        for (size_t xth = 0; xth < nx; xth++) {
            kernelFunc(
                cl::EnqueueArgs(
                    kernelRangeGlobal,
                    kernelRangeLocal
                ),
                yPlaneDev,
                static_cast<int>(xth),
                err
            );
        }
#endif
        cl::finish();
        // dump
        {
            auto yPlane = std::make_shared<std::vector<uint8_t>>(W * H);
            err |= cl::copy(yPlaneDev, yPlane->data(), yPlane->data() + yPlane->size());

            const auto file = DUMP_FILE + "_" + std::to_string(W) + "x" + std::to_string(H) + DUMP_FILE_EXT;
            std::ofstream dump(file, std::ios::binary);
            dump.write(reinterpret_cast<char*>(yPlane->data()), W * H);

            yPlane->assign(yPlane->size(), 128);
            dump.write(reinterpret_cast<char*>(yPlane->data()), (W / 2) * (H / 2));
            dump.write(reinterpret_cast<char*>(yPlane->data()), (W / 2) * (H / 2));
        }

        // Profiling
        // --------------------------------------------
        // warm up
        err |= kernel.setArg(0, yPlaneDev);
        err |= kernel.setArg(1, 0);
        for (int i = 0; i < 128; ++i) {
            err |= queue.enqueueNDRangeKernel(
                kernel,
                cl::NullRange,
                kernelRangeGlobal,
                kernelRangeLocal,
                NULL,
                NULL
            );
        }
        cl::finish();

        // use cl::Kernel for performance!
        cl::Event eventStart;
        cl::Event eventEnd;
#if SYNC_ON_DEV
        // Sync on Dev
        const int TIMES = 1000;
        for (int i = 0; i < TIMES; i++) {
            err |= queue.enqueueNDRangeKernel(
                kernel,
                cl::NullRange,
                kernelRangeGlobal,
                kernelRangeLocal,
                NULL,
                (i == 0) ? &eventStart : ((i == TIMES - 1) ? &eventEnd : NULL)
            );
        }
#else
        // Sync on Host
        const int TIMES = 10;
        for (int i = 0; i < TIMES; i++) {
            for (size_t xth = 0; xth < nx; xth++) {
                err |= kernel.setArg(1, static_cast<int>(xth));
                err |= queue.enqueueNDRangeKernel(
                    kernel,
                    cl::NullRange,
                    kernelRangeGlobal,
                    kernelRangeLocal,
                    NULL,
                    (i == 0 && xth == 0) ? &eventStart : ((i == TIMES - 1 && xth == nx - 1) ? &eventEnd : NULL)
                );
            }
        }
#endif
        err |= eventEnd.wait();

        cl_ulong start;
        cl_ulong end;
        err |= eventStart.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
        err |= eventEnd.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
        std::cout
        << "Kernel "
        << kernelName
        << "(): "
        << static_cast<double>(end - start) / 1000.0 / 1000.0
        << " [msec] / " << TIMES << " [times]"
        << std::endl;

        // Dump after 1000
        // --------------------------------------------
        {
            auto yPlane = std::make_shared<std::vector<uint8_t>>(W * H);
            err |= cl::copy(yPlaneDev, yPlane->data(), yPlane->data() + yPlane->size());

            const auto file = DUMP_FILE1000 + "_" + std::to_string(W) + "x" + std::to_string(H) + DUMP_FILE_EXT;
            std::ofstream dump(file, std::ios::binary);
            dump.write(reinterpret_cast<char*>(yPlane->data()), W * H);

            yPlane->assign(yPlane->size(), 128);
            dump.write(reinterpret_cast<char*>(yPlane->data()), (W / 2) * (H / 2));
            dump.write(reinterpret_cast<char*>(yPlane->data()), (W / 2) * (H / 2));
        }
    }
    catch (cl::Error err) {
        std::cerr
        << "ERROR: "
        << err.what()
        << "("
        << err.err()
        << ")"
        << std::endl;
    }

    return EXIT_SUCCESS;
}
