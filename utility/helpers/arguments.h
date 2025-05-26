#pragma once
#include <boost/program_options.hpp>
#include <chrono>
#include <iostream>
#include <string>
#include <math/math.h>
#include <math/template/metafunctions.h>
#include <filesystem>
#include <variant>

namespace arguments {



void loadbar(unsigned int x, unsigned int n, uint32_t w = 50, std::ostream &io = std::cout);

struct cmd {
  std::map<std::string, std::vector<std::vector<std::pair<float, float>>>> renderData;
  void enqueueRenderData(const std::map<std::string, std::vector<std::pair<float, float>>> &data);

  std::vector<std::string> jsons;
  
  std::string snapFile;
  bool snapped = false;

  static cmd &instance();
  bool headless;

  std::chrono::high_resolution_clock::time_point start;
  boost::program_options::variables_map vm;

  bool init(bool console, int argc, char *argv[]);

  void finalize();
  void logParameters();

  void parameter_stats();

  boost::program_options::variables_map &getVM();

  bool end_simulation_frame = false;
  int32_t timesteps = 50;

  bool end_simulation_time = false;
  double time_limit = 50.0;

  bool pause_simulation_time = false;
  double time_pause = 50.0;

  bool timers = true;
  bool rtx = false;

  bool renderToFile = false;
  std::filesystem::path renderDirectory = std::string("");
  bool outputSimulation = false;
  std::filesystem::path outputDirectory = std::string("");
};

} // namespace arguments
