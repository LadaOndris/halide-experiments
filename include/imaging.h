

#ifndef HALIDE_EXPERIMENTS_IMAGING_H
#define HALIDE_EXPERIMENTS_IMAGING_H

Buffer<uint8_t> createNoisyImage(int size, float gaussianNoiseSigma);

Buffer<uint8_t> loadImageFromFile(const std::string &filePath);

void saveImageToFile(Buffer<uint8_t> image, const std::string &targetFilePath);


#endif //HALIDE_EXPERIMENTS_IMAGING_H
