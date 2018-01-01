#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>
#include <streambuf>

static const size_t W = 1024;
static const size_t H = 1024;

const cl::NDRange kernelRangeGlobal(W, H);
const cl::NDRange kernelRangeLocal(16, 16);

#define kernelFile "src/pred.cl"
#define kernelName "pred"

static const std::string DUMP_FILE {"dump"};
static const std::string DUMP_FILE_EXT {".yuv"};


int main(void)
{
    cl_int err = CL_SUCCESS;
    try {
        // Platform
        // --------------------------------------------
        std::cout << "Platform" << std::endl;
        std::cout << "=======================================" << std::endl;
        std::vector<cl::Platform> platforms;
        err |= cl::Platform::get(&platforms);
        size_t n = platforms.size();
        for (size_t i = 0; i < n; i++) {
            std::cout << i << std::endl;
            std::cout << "---------------------------------------" << std::endl;

            static const int names[] = {
                CL_PLATFORM_PROFILE,
                CL_PLATFORM_VERSION,
                CL_PLATFORM_NAME,
                CL_PLATFORM_VENDOR,
                CL_PLATFORM_EXTENSIONS,
            };
            int n_names = sizeof(names) / sizeof(names[0]);
            for (int j = 0; j < n_names; j++) {
                std::string param;
                err |= platforms[i].getInfo(names[j], &param);
                std::cout << param << std::endl;
            }
        }
        if (n == 0) {
            std::cout << "Platform size 0\n";
            return -1;
        }

        // Context
        // --------------------------------------------
        cl_context_properties properties[] = {
            CL_CONTEXT_PLATFORM,
            (cl_context_properties)(platforms[0])(),
            0
        };
        cl::Context context(CL_DEVICE_TYPE_GPU, properties);

        // Device
        // --------------------------------------------
        std::cout << "Device" << std::endl;
        std::cout << "=======================================" << std::endl;
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
        size_t n_dev = devices.size();
        for (size_t i = 0; i < n_dev; i++) {
            std::cout << i << std::endl;
            std::cout << "---------------------------------------" << std::endl;

            static const int names[] = {
                CL_DEVICE_NAME,
                CL_DEVICE_VENDOR,
                CL_DEVICE_PROFILE,
                CL_DEVICE_VERSION,
                CL_DRIVER_VERSION,
                CL_DEVICE_OPENCL_C_VERSION,
                CL_DEVICE_EXTENSIONS,
            };
            int n_names = sizeof(names) / sizeof(names[0]);
            for (int j = 0; j < n_names; j++) {
                std::string param;
                err |= devices[i].getInfo(names[j], &param);
                std::cout << param << std::endl;
            }
        }

        // Build
        // --------------------------------------------
        std::ifstream from(kernelFile);
        std::string kernelStr((std::istreambuf_iterator<char>(from)),
                               std::istreambuf_iterator<char>());
        from.close();
        cl::Program::Sources sources(
            1,
            std::make_pair(kernelStr.c_str(), kernelStr.length())
        );
        cl::Program program_ = cl::Program(context, sources);
        err |= program_.build(devices);
        cl::Kernel kernel(program_, kernelName, &err);

        // Execution
        // --------------------------------------------
        cl::Buffer yPlaneDev(context, CL_MEM_READ_WRITE, W * H);
        err |= kernel.setArg(0, yPlaneDev);

        cl::CommandQueue queue(
            context,
            devices[0],
            CL_QUEUE_PROFILING_ENABLE, //0,
            &err);

        // warm up
        for (int i = 0; i < 128; ++i) {
            err |= queue.enqueueNDRangeKernel(
                kernel,
                cl::NullRange,
                kernelRangeGlobal,
                kernelRangeLocal,
                NULL,
                NULL);
        }
        cl::finish();

        const int TIMES = 1000;
        cl::Event eventStart;
        cl::Event eventEnd;
        err |= queue.enqueueNDRangeKernel(
            kernel,
            cl::NullRange,
            kernelRangeGlobal,
            kernelRangeLocal,
            NULL,
            &eventStart);
        for (int i = 0; i < TIMES - 2; ++i) {
            err |= queue.enqueueNDRangeKernel(
                kernel,
                cl::NullRange,
                kernelRangeGlobal,
                kernelRangeLocal,
                NULL,
                NULL);
        }
        err |= queue.enqueueNDRangeKernel(
            kernel,
            cl::NullRange,
            kernelRangeGlobal,
            kernelRangeLocal,
            NULL,
            &eventEnd);
        err |= eventEnd.wait();

        // Profiling
        // --------------------------------------------
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

        // Dump
        // --------------------------------------------
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
    }
    catch (cl::Error err) {
        std::cerr
        << "ERROR: "
        << err.what()
        << "("
        << err.err()
        << ")"
        << std::endl;

        if (err.err() == CL_BUILD_PROGRAM_FAILURE) {
            // TODO: build error
        }
    }

    return EXIT_SUCCESS;
}
