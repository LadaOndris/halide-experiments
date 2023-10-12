
#include "Halide.h"
#include <random>
#include <chrono>
#include <unistd.h> // for getopt

#include "lib/stb/stb_image.h"
#include "lib/stb/stb_image_write.h"
#include "pipelines/NonlocalMeansFilter.h"
#include "target.h"
#include "pipelines/ColorToGrayConverter.h"
#include "imaging.h"

using namespace Halide;

struct Arguments {
    std::string imagePath;
    std::string pipelineType;
    int reps = 1;
    std::string target;
    bool areValid = false;
};

Arguments processArguments(int argc, char **argv);

void processHalide(const std::string &imagePath, int reps,
                   const std::string &pipelineType, const std::string &targetType);

Target getTarget(const std::string &targetType);

std::shared_ptr<HalidePipeline> createPipeline(const std::string &pipelineType,
                                               const Buffer<uint8_t> &image);

Buffer<uint8_t> runPipeline(std::shared_ptr<HalidePipeline> pipeline,
                            const Buffer<uint8_t> &image,
                            const Target &target, int reps);

void printPipelineSchedule(const std::shared_ptr<HalidePipeline> &pipeline);

template<typename Func>
double measureExecutionTime(Func &&func);

void printCurrentTime();

int main(int argc, char **argv) {
    Arguments args = processArguments(argc, argv);
    if (!args.areValid) {
        return EXIT_FAILURE;
    }

    try {
        processHalide(args.imagePath, args.reps, args.pipelineType, args.target);
    } catch (CompileError &e) {
        std::cout << e.what() << std::endl;
    } catch (RuntimeError &e) {
        std::cout << e.what() << std::endl;
    }

    return EXIT_SUCCESS;
}

Arguments processArguments(int argc, char **argv) {
    Arguments args;
    int opt;
    while ((opt = getopt(argc, argv, "i:r:p:t:")) != -1) {
        switch (opt) {
            case 'i':
                args.imagePath = optarg;
                break;
            case 'r':
                args.reps = std::stoi(optarg);
                break;
            case 'p':
                args.pipelineType = optarg;
                break;
            case 't':
                args.target = optarg;
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " -i <image_path> -r <reps> -p <pipeline_type> -t <target>"
                          << std::endl;
                return args;
        }
    }
    if (args.imagePath.empty() || args.pipelineType.empty()) {
        std::cerr << "Both --image and --pipeline arguments are required." << std::endl;
        return args;
    }
    if (args.target != "cpu" && args.target != "gpu") {
        std::cerr << "--target (-t) must be one of [gpu, cpu]." << std::endl;
        return args;
    }
    args.areValid = true;
    return args;
}


void processHalide(const std::string &imagePath, int reps,
                   const std::string &pipelineType, const std::string &targetType) {
//    int imageSize = 20;
//    float gaussianNoiseSigma = 20.f;
//    auto image = createNoisyImage(imageSize, gaussianNoiseSigma);
    auto target = getTarget(targetType);

    std::cout << "Preparing input image..." << std::endl;
    auto image = loadImageFromFile(imagePath);
    saveImageToFile(image, "outputs/input.png");

    std::cout << "Instantiating pipeline..." << std::endl;
    auto pipeline = createPipeline(pipelineType, image);

    if (target.has_gpu_feature()) {
        std::cout << "Running pipeline on the GPU..." << std::endl;
        pipeline->scheduleForGPU();
    } else {
        std::cout << "Running pipeline on the CPU..." << std::endl;
        pipeline->scheduleForCPU();
    }
    printPipelineSchedule(pipeline);

    auto outputBuffer = runPipeline(pipeline, image, target, reps);

    std::cout << "Saving result..." << std::endl;
    saveImageToFile(outputBuffer, "outputs/output.png");
}

Target getTarget(const std::string &targetType) {
    Target target;
    if (targetType == "gpu") {
        std::cout << "Searching for a GPU target..." << std::endl;
        target = find_gpu_target();
    } else if (targetType == "cpu") {
        std::cout << "Searching for a CPU target..." << std::endl;
        target = get_host_target();
    } else {
        std::cerr << "Unknown target type: " << targetType << std::endl;
    }

    std::cout << "The target found: " << target.to_string().c_str() << std::endl;
    return target;
}


std::shared_ptr<HalidePipeline> createPipeline(const std::string &pipelineType,
                                               const Buffer<uint8_t> &image) {
    int searchWindowSize = 13;
    int patchSize = 5;

    std::shared_ptr<HalidePipeline> pipeline;
    if (pipelineType == "colortogray") {
        assert(image.channels() == 3);
        pipeline = std::make_shared<ColorToGrayConverter>(image);
    } else if (pipelineType == "nonlocalmeans") {
        assert(image.channels() == 1);
        pipeline = std::make_shared<NonlocalMeansFilter>(image, patchSize, searchWindowSize);
    } else {
        std::cerr << "Invalid pipeline type: " << pipelineType << std::endl;
        return nullptr;
    }
    return pipeline;
}

Buffer<uint8_t> runPipeline(std::shared_ptr<HalidePipeline> pipeline,
                            const Buffer<uint8_t> &image,
                            const Target &target, int reps) {
    auto realizationWidth = image.width();
    auto realizationHeight = image.height();

    auto outputBuffer = Halide::Buffer<uint8_t>(realizationWidth, realizationHeight);

    double warmupTime = measureExecutionTime([&pipeline, &outputBuffer, &target] {
        // Warm-up before measuring
        pipeline->result.realize(outputBuffer, target);

        // Copy from GPU. Must be called, because the GPU runs asynchronously.
        if (target.has_gpu_feature()) {
            outputBuffer.copy_to_host();
        }
    });
    double executionTime = measureExecutionTime([&pipeline, &outputBuffer, &reps, &target] {
        for (int i = 0; i < reps; i++) {
            pipeline->result.realize(outputBuffer, target);

            // Copy from GPU. Must be done for each rep, because the GPU runs asynchronously.
            if (target.has_gpu_feature()) {
                outputBuffer.copy_to_host();
            }
        }
    });

    std::cout << "Warmup time: " << warmupTime * 1000 << " ms" << std::endl;
    std::cout << "Execution time: " << (executionTime / reps) * 1000 << " ms/rep" << std::endl;

    return outputBuffer;
}

void printPipelineSchedule(const std::shared_ptr<HalidePipeline> &pipeline) {
    printf("\nPseudo-code for the schedule:\n");
    pipeline->result.print_loop_nest();
    printf("\n");
}

// Custom timing function
template<typename Func>
double measureExecutionTime(Func &&func) {
    using namespace std::chrono;

    // Start the timer
    high_resolution_clock::time_point start_time = high_resolution_clock::now();

    // Execute the provided function or code block
    func();

    // Stop the timer
    high_resolution_clock::time_point end_time = high_resolution_clock::now();

    // Calculate the elapsed time in seconds
    duration<double> elapsed_seconds = duration_cast<duration<double>>(end_time - start_time);

    return elapsed_seconds.count();
}

void printCurrentTime() {
    // Capture the current time point
    auto currentTimePoint = std::chrono::high_resolution_clock::now();

    // Convert the time point to a time_t (C time) representation
    std::time_t time = std::chrono::high_resolution_clock::to_time_t(currentTimePoint);

    // Convert the time_t to a string for printing
    std::string currentTimeString = std::ctime(&time);

    // Print the current time
    std::cout << "Current time: " << currentTimeString;
}

