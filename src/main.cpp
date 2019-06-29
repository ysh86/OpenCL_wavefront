#if 1
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define SYNC_ON_DEV 1
#else
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#define SYNC_ON_DEV 1
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
cl::NDRange kernelRangeGlobal(16, 16);
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
        // Platforms
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

        cl::Platform &plat = platforms[PLATFORM_INDEX];
        std::cout << "Use platform " << PLATFORM_INDEX << std::endl;
        std::cout << std::endl;


        // Devices & Context
        // --------------------------------------------
        std::cout << "=======================================" << std::endl;
        std::cout << "Devices" << std::endl;
        std::cout << "=======================================" << std::endl;
        std::vector<cl::Device> devices;
        err |= plat.getDevices(CL_DEVICE_TYPE_ALL, &devices);
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
        if (devices.size() == 0) {
            std::cout << "ERROR: No devices" << std::endl;
            return EXIT_FAILURE;
        }
        std::cout << std::endl;

        i = 0;
        cl::Device &device = devices[i];
        std::cout << "Use device " << i << std::endl;
        std::cout << std::endl;
        // Context
        cl::Context context(device);


        // Build
        // --------------------------------------------
        std::ifstream from(kernelFile);
        std::string kernelStr((std::istreambuf_iterator<char>(from)),
                               std::istreambuf_iterator<char>());
        from.close();
        cl::Program::Sources sources {kernelStr};
        cl::Program program = cl::Program(context, sources);
        try {
#if CL_HPP_TARGET_OPENCL_VERSION >= 200
            err |= program.build("-cl-std=CL2.0");
#else
            err |= program.build("");
#endif
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
            context,
            cl::QueueProperties::Profiling //cl::QueueProperties::None
        );

#if SYNC_ON_DEV
        auto kernelFunc = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer>(program, kernelName);
#else
        auto kernelFunc = cl::KernelFunctor<cl::Buffer, int, int>(program, kernelName);
#endif
        auto kernel = kernelFunc.getKernel();
        size_t s = kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device);
        std::cout << "CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: " << s << std::endl;

        cl::Buffer yPlaneDev(context, CL_MEM_READ_WRITE, W * H);
#if SYNC_ON_DEV
        // Sync on Dev
        const size_t condsSize = sizeof(int32_t) * (W / 16) * (H / 16);
        cl::Buffer condsDev(context, CL_MEM_READ_WRITE, condsSize);

        // diagonal order
        int32_t orders[(W/16)*(H/16)];
        // -> xth
        // 0(0<<1): 0 1 2 3 4 5 6 7
        // 1(1<<1):     2 3 4 5 6 7 8 9
        // 2(2<<1):         4 5 6 7 8 9 10 11
        size_t nx = W / 16 + ((H / 16 - 1) * 2);
        size_t cur = 0;
        for (size_t xth = 0; xth < nx; xth++) {
            size_t offsetY = (xth < W / 16) ? 0 : (xth - W / 16) / 2 + 1;
            size_t ny = (xth >> 1) + 1;
            ny = (ny < H / 16) ? ny - offsetY : H / 16 - offsetY;

            for (size_t groupY = offsetY; groupY < offsetY + ny; groupY++) {
                size_t groupX = xth - (groupY << 1);
                orders[cur++] = (W / 16) * groupY + groupX;
            }
        }
        cl::Buffer ordersDev(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, condsSize, orders);

        const uint8_t patternZero = 0;
        err |= queue.enqueueFillBuffer(condsDev, patternZero, 0, condsSize, NULL, NULL);
        kernelFunc(
            cl::EnqueueArgs(
                queue,
                kernelRangeGlobal,
                kernelRangeLocal
            ),
            yPlaneDev,
            condsDev,
            ordersDev,
            err
        );
#else
        // Sync on Host
        // -> xth
        // 0(0<<1): 0 1 2 3 4 5 6 7
        // 1(1<<1):     2 3 4 5 6 7 8 9
        // 2(2<<1):         4 5 6 7 8 9 10 11
        size_t nx = W / 16 + ((H / 16 - 1) * 2);
        for (size_t xth = 0; xth < nx; xth++) {
            size_t offsetY = (xth < W / 16) ? 0 : (xth - W / 16) / 2 + 1;
            size_t ny = (xth >> 1) + 1;
            ny = (ny < H / 16) ? ny - offsetY : H / 16 - offsetY;
            kernelRangeGlobal.get()[1] = ny * 16;
            kernelFunc(
                cl::EnqueueArgs(
                    queue,
                    kernelRangeGlobal,
                    kernelRangeLocal
                ),
                yPlaneDev,
                static_cast<int>(xth),
                static_cast<int>(offsetY),
                err
            );
        }
#endif
        queue.finish();
        // dump
        {
            auto yPlane = std::make_shared<std::vector<uint8_t>>(W * H);
            err |= cl::copy(queue, yPlaneDev, yPlane->data(), yPlane->data() + yPlane->size());

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
#if SYNC_ON_DEV
        err |= kernel.setArg(1, condsDev);
        err |= kernel.setArg(2, ordersDev);
#else
        err |= kernel.setArg(1, 0);
        err |= kernel.setArg(2, 0);
#endif
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
        queue.finish();

        // use cl::Kernel for performance!
        cl::Event eventStart;
        cl::Event eventEnd;
        const int TIMES = 1000;
#if SYNC_ON_DEV
        // Sync on Dev
        for (int i = 0; i < TIMES; i++) {
            err |= queue.enqueueFillBuffer(
                condsDev,
                patternZero,
                0,
                condsSize,
                NULL,
                /*(i == 0) ? &eventStart :*/ NULL // TODO: ???
            );
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
        for (int i = 0; i < TIMES; i++) {
            for (size_t xth = 0; xth < nx; xth++) {
                size_t offsetY = (xth < W / 16) ? 0 : (xth - W / 16) / 2 + 1;
                size_t ny = (xth >> 1) + 1;
                ny = (ny < H / 16) ? ny - offsetY : H / 16 - offsetY;
                kernelRangeGlobal.get()[1] = ny * 16;
                err |= kernel.setArg(1, static_cast<int>(xth));
                err |= kernel.setArg(2, static_cast<int>(offsetY));
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
            err |= cl::copy(queue, yPlaneDev, yPlane->data(), yPlane->data() + yPlane->size());

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
