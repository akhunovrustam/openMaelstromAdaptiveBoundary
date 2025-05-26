#include <iostream>
#include <vector>
//#define BOOST_MSVC
// using std::is_assignable;
// using std::is_volatile;
#include <IO/config/config.h>
#include <IO/config/parser.h>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/case_conv.hpp>
#include <boost/optional.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/type_traits/is_assignable.hpp>
#include <boost/type_traits/is_volatile.hpp>

#include <boost/format.hpp>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <istream>
#include <iterator>
#include <random>
#include <sstream>
#include <thread>
#include <utility/MemoryManager.h>
#include <utility/helpers/arguments.h>
#include <tools/log.h>
#include <tools/pathfinder.h>
#include <tools/timer.h>
#include <utility/identifier/arrays.h>
#include <utility/identifier/uniform.h>
#include <utility/macro.h>
#include <math/template/tuple_for_each.h>

thread_local int3 h_threadIdx;
thread_local int3 h_blockDim;
thread_local int3 h_blockIdx;

struct param_stats_base {
  virtual std::string to_string(size_t field_len = 5) = 0;
  virtual void sample() = 0;
  virtual std::string name() = 0;
};
std::string pad(std::string str, const size_t num = 9, const char paddingChar = ' ') {
  if (num > str.size())
    str.insert(0, num - str.size(), paddingChar);
  if (str.size() > num) {
    str = str.substr(0, num);
  }
  return str;
}

template <typename param> struct param_stats : public param_stats_base {
  using T = typename param::type;

  std::vector<T> samples;

  std::string to_string(size_t field_len = 5) {
    if (samples.size() == 0)
      return "";
    std::vector<T> sam = samples;
    std::nth_element(sam.begin(), sam.begin() + sam.size() / 2, sam.end(),
                     [](auto lhs, auto rhs) { return lhs < rhs; });
    auto median = sam[sam.size() / 2];

    T min = samples[0];
    T max = samples[0];
    T avg{};
    double ctr = static_cast<double>(samples.size());

    for (auto s : samples) {
      min = math::min(s, min);
      max = math::max(s, max);
      avg += s;
    }
    avg /= static_cast<T>(ctr);
    T stddev{};
    for (auto s : samples) {
      auto diff = (s - avg) * (s - avg);
      stddev += diff;
    }
    stddev /= static_cast<T>(ctr) - static_cast<T>(1.0);
    stddev = static_cast<T>(sqrt(stddev));

    std::stringstream sstream;
    sstream << std::setw(field_len) << param::jsonName;
    sstream << " ";
    sstream << pad(IO::config::convertToString(avg)) << " ";
    sstream << pad(IO::config::convertToString(median)) << " ";
    sstream << pad(IO::config::convertToString(min)) << " ";
    sstream << pad(IO::config::convertToString(max)) << " ";
    sstream << pad(IO::config::convertToString(stddev)) << " ";

    // sstream << std::endl;
    return sstream.str();
  }

  void sample() { samples.push_back(get<param>()); }

  std::string name() { return param::jsonName; }
};
std::vector<param_stats_base *> param_watchlist;

static std::vector<int32_t> numPtcls, blendedPtcls, densityIterations, divergenceIterations, subSteps;
static std::vector<int32_t> sharedPtcls, mergedPtcls, splitPtcls;
static std::vector<float> densityErrors, divergenceErrors, maxVelocities, timeSteps, simulationTimes;
static std::vector<float> frameTimes, densityTimes, iTimes, dfTimes, constrainTimes, neighborTimes, resortTimes,
    adaptiveTimes;
static Timer *frameTimer, *densityTimer, *iTimer, *dfTimer, *constrainTimer, *neighborTimer, *resortTimer,
    *adaptiveTimer;

void arguments::cmd::logParameters() {
  static bool once = true;
  if (once) {
    auto &timers = TimerManager::getTimers();
    for (auto t : timers) {
      frameTimer = (t->getDecriptor() == "Frame" && frameTimer == nullptr) ? t : frameTimer;
      densityTimer = (t->getDecriptor() == "Density estimate" && densityTimer == nullptr) ? t : densityTimer;
      iTimer = (t->getDecriptor() == "DFSPH: incompressibility" && iTimer == nullptr) ? t : iTimer;
      dfTimer = (t->getDecriptor() == "DFSPH: divergence solver" && dfTimer == nullptr) ? t : dfTimer;
      constrainTimer = (t->getDecriptor() == "Constraining Support" && constrainTimer == nullptr) ? t : constrainTimer;
      neighborTimer =
          (t->getDecriptor() == "Constrained Neighborlist creation" && neighborTimer == nullptr) ? t : neighborTimer;
      resortTimer = (t->getDecriptor() == "compactMLM sort: resort" && resortTimer == nullptr) ? t : resortTimer;
      adaptiveTimer = (t->getDecriptor() == "Adaptivity: adapt" && adaptiveTimer == nullptr) ? t : adaptiveTimer;
    }
    once = false;
  }
  numPtcls.push_back(get<parameters::internal::num_ptcls>());
  subSteps.push_back(1);
  blendedPtcls.push_back(get<parameters::adaptive::blendedPtcls>());
  densityIterations.push_back(get<parameters::dfsph_settings::densitySolverIterations>());
  divergenceIterations.push_back(get<parameters::dfsph_settings::divergenceSolverIterations>());
  simulationTimes.push_back(get<parameters::internal::simulationTime>());

  sharedPtcls.push_back(
      std::accumulate(get<parameters::adaptive::sharedPtcls>().begin(), get<parameters::adaptive::sharedPtcls>().end(), 0));
  mergedPtcls.push_back(
      std::accumulate(get<parameters::adaptive::mergedPtcls>().begin(), get<parameters::adaptive::mergedPtcls>().end(), 0));
  splitPtcls.push_back(std::accumulate(get<parameters::adaptive::splitPtcls>().begin(), get<parameters::adaptive::splitPtcls>().end(), 0));

  static auto time = 0.0;
  timeSteps.push_back(get<parameters::internal::simulationTime>() - time);
  time = get<parameters::internal::simulationTime>();
  densityErrors.push_back(get<parameters::dfsph_settings::densityError>());
  divergenceErrors.push_back(get<parameters::dfsph_settings::divergenceError>());
  maxVelocities.push_back(get<parameters::internal::max_velocity>());
  if (frameTimer->getSamples().size() > 0) {
    frameTimes.push_back(frameTimer->getSamples()[frameTimer->getSamples().size() - 1]);
    densityTimes.push_back(densityTimer->getSamples()[densityTimer->getSamples().size() - 1]);
    /*iTimes.push_back(iTimer->getSamples()[iTimer->getSamples().size() - 1]);
    dfTimes.push_back(dfTimer->getSamples()[dfTimer->getSamples().size() - 1]);
    */resortTimes.push_back(resortTimer->getSamples()[resortTimer->getSamples().size() - 1]);
    if (adaptiveTimer != nullptr)
      adaptiveTimes.push_back(adaptiveTimer->getSamples()[adaptiveTimer->getSamples().size() - 1]);
    else
      adaptiveTimes.push_back(0.0);

  } else {
    frameTimes.push_back(0.0);
    densityTimes.push_back(0.0);
    iTimes.push_back(0.0);
    dfTimes.push_back(0.0);
    resortTimes.push_back(0.0);
    adaptiveTimes.push_back(0.0);
  }
  if (neighborTimer != nullptr && neighborTimer->getSamples().size() > 0) {
    constrainTimes.push_back(constrainTimer->getSamples()[constrainTimer->getSamples().size() - 1]);
    neighborTimes.push_back(neighborTimer->getSamples()[neighborTimer->getSamples().size() - 1]);
  } else {
    constrainTimes.push_back(0.0);
    neighborTimes.push_back(0.0);
  }
}

void printLoggedParameters() {
  auto pivotData = [](auto v) {
    auto t_l = 0.0;
    std::vector<double> sums, avgs;
    auto weight = 0.0;
    auto sum = 0.0;
    auto ts = 0.0;
    for (int32_t i = 0; i < simulationTimes.size(); ++i) {
      auto t_c = ts;
      auto t_n = simulationTimes[i];
      ts = t_n;
      double a0 = math::clamp((t_l - t_c) / (t_n - t_c), 0.0, 1.0);
      double a1 = math::clamp((t_l + 1.0 / 60.0 - t_c) / (t_n - t_c), 0.0, 1.0);
      double a = a1 - a0;
      sum += v[i] * a;
      weight += a;
      // std::cout << t_c << " -> " << t_n << " / " << t_l << " @" << a << "[" << a0 << "/" << a1 << "]: " << v[i] << "
      // - " << sums.size() << std::endl;
      if (t_n > t_l + 1.0 / 60.0) {
        t_l += 1.0 / 60.0;
        sums.push_back(sum);
        avgs.push_back(sum / weight);
        sum = v[i] * (1.0 - a);
        weight = 1.0 - a;
      }
    }
    return std::make_pair(sums, avgs);
  };
  // static std::vector<int32_t> numPtcls, blendedPtcls, densityIterations, divergenceIterations;
  // static std::vector<int32_t> sharedPtcls, mergedPtcls, splitPtcls;
  // static std::vector<float> densityErrors, divergenceErrors, maxVelocities, timeSteps, simulationTimes;
  // static std::vector<float> frameTimes, densityTimes, iTimes, dfTimes, constrainTimes, neighborTimes, resortTimes;
  if (numPtcls.size() == 0)
    return;
  auto [sum_numPtcls, avg_numPtcls] = pivotData(numPtcls);
  auto [sum_subSteps, avg_subSteps] = pivotData(subSteps);
  auto [sum_blendedPtcls, avg_blendedPtcls] = pivotData(blendedPtcls);
  auto [sum_densityIterations, avg_densityIterations] = pivotData(densityIterations);
  auto [sum_divergenceIterations, avg_divergenceIterations] = pivotData(divergenceIterations);
  auto [sum_sharedPtcls, avg_sharedPtcls] = pivotData(sharedPtcls);
  auto [sum_mergedPtcls, avg_mergedPtcls] = pivotData(mergedPtcls);
  auto [sum_splitPtcls, avg_splitPtcls] = pivotData(splitPtcls);
  auto [sum_densityErrors, avg_densityErrors] = pivotData(densityErrors);
  auto [sum_divergenceErrors, avg_divergenceErrors] = pivotData(divergenceErrors);
  auto [sum_maxVelocities, avg_maxVelocities] = pivotData(maxVelocities);
  auto [sum_timeSteps, avg_timeSteps] = pivotData(timeSteps);
  auto [sum_simulationTimes, avg_simulationTimes] = pivotData(simulationTimes);
  auto [sum_frameTimes, avg_frameTimes] = pivotData(frameTimes);
  auto [sum_densityTimes, avg_densityTimes] = pivotData(densityTimes);
  auto [sum_iTimes, avg_iTimes] = pivotData(iTimes);
  auto [sum_dfTimes, avg_dfTimes] = pivotData(dfTimes);
  auto [sum_constrainTimes, avg_constrainTimes] = pivotData(constrainTimes);
  auto [sum_neighborTimes, avg_neighborTimes] = pivotData(neighborTimes);
  auto [sum_resortTimes, avg_resortTimes] = pivotData(resortTimes);
  auto [sum_adaptiveTimes, avg_adaptiveTimes] = pivotData(adaptiveTimes);

  auto removeOutlier = [](auto v) {
    auto idb = static_cast<int32_t>(::ceil(static_cast<double>(v.size()) * 0.025));
    auto ide = static_cast<int32_t>(::floor(static_cast<double>(v.size()) * 0.975));
    std::sort(v.begin(), v.end());
    std::vector<double> c(v.begin() + idb, v.begin() + ide);
    return c;
  };

  auto c_numPtcls = removeOutlier(avg_numPtcls);
  auto c_subSteps = removeOutlier(sum_subSteps);
  auto c_blendedPtcls = removeOutlier(sum_blendedPtcls);
  auto c_densityIterations = removeOutlier(avg_densityIterations);
  auto c_divergenceIterations = removeOutlier(avg_divergenceIterations);
  auto c_sharedPtcls = removeOutlier(sum_sharedPtcls);
  auto c_mergedPtcls = removeOutlier(sum_mergedPtcls);
  auto c_splitPtcls = removeOutlier(sum_splitPtcls);
  auto c_densityErrors = removeOutlier(avg_densityErrors);
  auto c_divergenceErrors = removeOutlier(avg_divergenceErrors);
  auto c_maxVelocities = removeOutlier(avg_maxVelocities);
  auto c_timeSteps = removeOutlier(avg_timeSteps);
  auto c_simulationTimes = removeOutlier(avg_simulationTimes);
  auto c_frameTimes = removeOutlier(sum_frameTimes);
  auto c_densityTimes = removeOutlier(sum_densityTimes);
  auto c_iTimes = removeOutlier(sum_iTimes);
  auto c_dfTimes = removeOutlier(sum_dfTimes);
  auto c_constrainTimes = removeOutlier(sum_constrainTimes);
  auto c_neighborTimes = removeOutlier(sum_neighborTimes);
  auto c_resortTimes = removeOutlier(sum_resortTimes);
  auto c_adaptiveTimes = removeOutlier(sum_adaptiveTimes);

  int32_t number_field = 10;
  int32_t name_field = 24;
  auto vStats = [&](auto v) {
    if (v.size() == 0)
      return std::string("");
    std::vector<double> sam;
    for (auto ve : v)
      sam.push_back(static_cast<double>(ve));
    std::nth_element(sam.begin(), sam.begin() + sam.size() / 2, sam.end(),
                     [](auto lhs, auto rhs) { return lhs < rhs; });
    auto median = sam[sam.size() / 2];

    auto min = sam[0];
    auto max = sam[0];
    auto avg = decltype(min){};
    double ctr = static_cast<double>(sam.size());

    for (auto s : sam) {
      min = math::min(s, min);
      max = math::max(s, max);
      avg += s;
    }
    avg /= static_cast<double>(ctr);
    double stddev{};
    for (auto s : sam) {
      auto diff = (s - avg) * (s - avg);
      stddev += diff;
    }
    stddev /= static_cast<double>(ctr) - static_cast<double>(1.0);
    stddev = static_cast<double>(sqrt(stddev));

    std::stringstream sstream;

    auto fmtFloat = [=](double v) {
      std::stringstream sstream;
      if (v >= 0.0)
        sstream << " " << std::setw(number_field - 1);
      else
        sstream << std::setw(number_field);
      if (::abs(v) < 1e5 && ::abs(v) > 1e-4)
        sstream << std::fixed << std::showpoint << std::setprecision(7 - (int32_t)math::max(::floor(::log10(v)), 0.0))
                << v;
      else
        sstream << std::scientific << std::setprecision(3) << v;
      return sstream.str();
    };

    sstream << std::setw(number_field) << std::scientific << std::setprecision(5) << std::showpos;
    sstream << " ";

    sstream << fmtFloat(avg) << " ";
    sstream << fmtFloat(median) << " ";
    sstream << fmtFloat(min) << " ";
    sstream << fmtFloat(max) << " ";
    sstream << fmtFloat(stddev) << " ";
    return sstream.str();
  };

  std::cout << std::endl;
  std::cout << std::setw(name_field + 3) << "Name"
            << "    " << std::setw(number_field + 2) << "avg    " << std::setw(number_field + 2) << "med    "
            << std::setw(number_field + 2) << "min    " << std::setw(number_field + 2) << "max    "
            << std::setw(number_field + 2) << "dev    " << std::endl;
  std::cout << "Simulation statistics: " << std::endl;
  std::cout << std::setw(name_field) << "numPtcls" << vStats(c_numPtcls) << std::endl;
  std::cout << std::setw(name_field) << "subSteps" << vStats(c_subSteps) << std::endl;
  std::cout << std::setw(name_field) << "maxVelocities" << vStats(c_maxVelocities) << std::endl;
  std::cout << std::setw(name_field) << "timeSteps" << vStats(c_timeSteps) << std::endl;

  std::cout << "Solver statistics: " << std::endl;
  std::cout << std::setw(name_field) << "densityErrors" << vStats(c_densityErrors) << std::endl;
  std::cout << std::setw(name_field) << "densityIterations" << vStats(c_densityIterations) << std::endl;
  std::cout << std::setw(name_field) << "divergenceErrors" << vStats(c_divergenceErrors) << std::endl;
  std::cout << std::setw(name_field) << "divergenceIterations" << vStats(c_divergenceIterations) << std::endl;

  std::cout << "Timing statistics: " << std::endl;
  std::cout << std::setw(name_field) << "frameTimes" << vStats(c_frameTimes) << std::endl;
  std::cout << std::setw(name_field) << "densityTimes" << vStats(c_densityTimes) << std::endl;
  std::cout << std::setw(name_field) << "iTimes" << vStats(c_iTimes) << std::endl;
  std::cout << std::setw(name_field) << "dfTimes" << vStats(c_dfTimes) << std::endl;
  std::cout << std::setw(name_field) << "constrainTimes" << vStats(c_constrainTimes) << std::endl;
  std::cout << std::setw(name_field) << "neighborTimes" << vStats(c_neighborTimes) << std::endl;
  std::cout << std::setw(name_field) << "resortTimes" << vStats(c_resortTimes) << std::endl;

  if (get<parameters::modules::adaptive>() == true) {
    std::cout << "Adaptivity statistics: " << std::endl;
    std::cout << std::setw(name_field) << "blendedPtcls" << vStats(c_blendedPtcls) << std::endl;
    std::cout << std::setw(name_field) << "sharedPtcls" << vStats(c_sharedPtcls) << std::endl;
    std::cout << std::setw(name_field) << "mergedPtcls" << vStats(c_mergedPtcls) << std::endl;
    std::cout << std::setw(name_field) << "splitPtcls" << vStats(c_splitPtcls) << std::endl;
    std::cout << std::setw(name_field) << "Timing" << vStats(c_adaptiveTimes) << std::endl;
  }
}

void arguments::loadbar(unsigned int x, unsigned int n, uint32_t w, std::ostream &io) {
  if ((x != n) && (x % (n / 100 + 1) != 0))
    return;

  float ratio = x / (float)n;
  uint32_t c = static_cast<uint32_t>(ratio * w);

  io << std::setw(3) << (uint32_t)(ratio * 100) << "% [";
  for (uint32_t x = 0; x < c; x++)
    io << "=";
  for (uint32_t x = c; x < w; x++)
    io << " ";
  io << "]";

  static auto start_time = std::chrono::high_resolution_clock::now();

  auto current_time = std::chrono::high_resolution_clock::now();

  io << "\r" << formatFloat((float)std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count());

  float tpp = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count() / ratio;

  io << "s of est. " << formatFloat(tpp) << "s" << std::flush;
}

arguments::cmd &arguments::cmd::instance() {
  static cmd c;
  return c;
}
bool arguments::cmd::init(bool console, int argc, char *argv[]) {
  headless = console;
  start = std::chrono::high_resolution_clock::now();
  namespace po = boost::program_options;
  po::options_description desc("Allowed options");
  desc.add_options()
      ("output,o", po::value<std::string>(), "...")
      ("snap,s", po::value<std::string>(), "produce help message")
      ("snap,s", po::value<std::string>(), "produce help message")(
      "set", po::value<std::string>(), "produce help message")("device,d", po::value<int32_t>(), "set CUDA device")(
      "params,p", "watch parameters throughout the simulation")("help,h", "produce help message")(
      "frames,f", po::value<int>(), "set frame limiter")("time,t", po::value<double>(), "set time limiter")(
      "pause", po::value<double>(), "set time limiter")("verbose,v", "louden the logger on the console")(
      "no-timer,nt", "silence the timer output on the console")("memory,mem", "print memory consumption at end")(
      "info,i", "print total runtime at end")("log", po::value<std::string>(), "log to file")(
      "rtx", po::value<int>()->implicit_value(0), "enable ray Tracing from the start")(
      "config", po::value<std::string>(), "specify config file to open")("config_id,c", po::value<int>(),
                                                                         "load known config with given id")
      ("list,l", "list all known ids")(
      "params,p", "watch parameters and print them")("record,r", po::value<std::string>(), "log to file")(
      "render", po::value<std::string>(), "render to directory")("config_log", po::value<std::string>(),
                                                                 "save the config at the end to file")(
      "neighbordump", po::value<std::string>(), "writes a copy of the neighborlist at the end to file")(
      "json,j", po::value<std::vector<std::string>>(&jsons)->multitoken(),
      "writes a copy of the neighborlist at the end to file")("deviceRegex,G", po::value<std::string>(),
                                                              "regex to force functions to be called on device")(
      "hostRegex,H", po::value<std::string>(), "regex to force functions to be called on host")(
      "debugRegex,D", po::value<std::string>(), "regex to force functions to be called in debug mode");

  po::store(po::command_line_parser(argc, argv).options(desc).allow_unregistered().run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    std::cout << std::endl;
    std::cout << "modules usable for simulation ( use -j modules.name=option to configure them)" << std::endl;
    std::cout << "modules.sorting = hashed_cell|linear_cell" << std::endl;
    std::cout << "modules.hash_width = 32bit|64bit" << std::endl;
    std::cout << "modules.neighborhood = constrained|cell_based" << std::endl;
    std::cout << std::endl;
    std::cout << "modules.pressure = IISPH|IISPH17|DFSPH" << std::endl;
    std::cout << "modules.drag = Gissler17" << std::endl;
    std::cout << "modules.tension = Akinci" << std::endl;
    std::cout << "modules.vorticity = Bender17" << std::endl;
    std::cout << std::endl;
    std::cout << "modules.error_checking = true|false" << std::endl;
    std::cout << "modules.gl_record = true|false" << std::endl;
    std::cout << "modules.alembic_export = true|false" << std::endl;
    std::cout << std::endl;
    std::cout << "modules.volumeBoundary = true|false" << std::endl;
    std::cout << "modules.xsph = true|false" << std::endl;
    std::cout << "modules.surfaceDistance = true|false" << std::endl;
    std::cout << "modules.sprayAndFoam = true|false" << std::endl;
    std::cout << "modules.adaptive = true|false" << std::endl;
    std::cout << "modules.viscosity = true|false" << std::endl;

    return false;
  }

  logger::silent = true;
  if (vm.count("device")) {
    cudaSetDevice(vm["device"].as<int32_t>());
  }

  //if (vm.count("params")) {
  //  param_watchlist.push_back(new param_stats<parameters::internal::num_ptcls>());
  //  param_watchlist.push_back(new param_stats<parameters::internal::timestep>());
  //  param_watchlist.push_back(new param_stats<parameters::iisph_settings::iterations>());
  //  param_watchlist.push_back(new param_stats<parameters::iisph_settings::density_error>());
  //}
  if (vm.count("set")) {
    std::stringstream sstream;
    sstream << "particleSets.set1=" << vm["set"].as<std::string>();
    jsons.push_back(sstream.str());
  }

  if (vm.count("verbose"))
    logger::silent = false;
  if (vm.count("record")) {
    jsons.push_back("modules.gl_record=true");
    std::stringstream sstream;
    sstream << "render_settings.gl_file=\"" << vm["record"].as<std::string>() << "\"";
    jsons.push_back(sstream.str());
  }
  if (vm.count("no-timer"))
    timers = false;
  if (vm.count("frames")) {
    timesteps = vm["frames"].as<int>();
    end_simulation_frame = true;
  }
  if (vm.count("time")) {
    time_limit = vm["time"].as<double>();
    end_simulation_time = true;
  }
  if (vm.count("pause")) {
    time_pause = vm["pause"].as<double>();
    pause_simulation_time = true;
  }
  if (vm.count("deviceRegex")) {
    std::stringstream sstream;
    sstream << "simulation_settings.deviceRegex=" << vm["deviceRegex"].as<std::string>();
    jsons.push_back(sstream.str());
  }
  if (vm.count("hostRegex")) {
    std::stringstream sstream;
    sstream << "simulation_settings.hostRegex=" << vm["hostRegex"].as<std::string>();
    jsons.push_back(sstream.str());
  }
  if (vm.count("debugRegex")) {
    std::stringstream sstream;
    sstream << "simulation_settings.debugRegex=" << vm["debugRegex"].as<std::string>();
    jsons.push_back(sstream.str());
  }
  if (vm.count("render")) {
    renderToFile = true;
    renderDirectory = std::filesystem::path(get<parameters::internal::working_directory>());
    // std::cout << renderDirectory.string() << std::endl;
    renderDirectory /= vm["render"].as<std::string>();
    // std::cout << renderDirectory.string() << std::endl;
    if (!std::filesystem::exists(renderDirectory))
      std::filesystem::create_directories(renderDirectory);
  }

  if (vm.count("rtx")) {
    jsons.push_back("modules.rayTracing=true");
    jsons.push_back(std::string("modules.renderMode=") + std::to_string(vm["rtx"].as<int32_t>()));
    rtx = true;
  }

  auto config_paths = resolveFile("cfg/paths.cfg");
  std::ifstream ifs(config_paths.string());
  std::vector<std::string> config_folders;
  std::copy(std::istream_iterator<std::string>(ifs), std::istream_iterator<std::string>(),
            std::back_inserter(config_folders));
  ifs.close();
  std::filesystem::path default_config = "";
  try {
    default_config = resolveFile("OutletTest/config.yaml", config_folders);
    get<parameters::internal::config_file>() = default_config.string();
  } catch (...) {
    std::cout << "Could not find default configuration" << std::endl;
  }

  if (vm.count("snap")) {
    snapFile = resolveFile(vm["snap"].as<std::string>(), config_folders).string();
    snapped = true;
  }
  if (vm.count("output")) {
      outputSimulation = true;
      outputDirectory = vm["output"].as<std::string>();
  }
  // snapFile = resolveFile(R"(C:\Users\Winchenbach\adaptive\splash_001\frame_300.dump)", config_folders).string();
  // snapped = true;

  if (vm.count("config")) {
    get<parameters::internal::config_file>() = resolveFile(vm["config"].as<std::string>(), config_folders).string();
    std::ofstream file;
    auto path = resolveFile("cfg/configs.sph");
    file.open(resolveFile("cfg/configs.sph").string(), std::ios_base::app);
    file << vm["config"].as<std::string>() << std::endl;
    file.close();
  }
  if (vm.count("list")) {
    std::ifstream file(resolveFile("cfg/configs.sph").string());
    std::vector<std::string> Configs;
    std::copy(std::istream_iterator<std::string>(file), std::istream_iterator<std::string>(),
              std::back_inserter(Configs));
    int32_t i = 0;
    for (auto c : Configs) {
      std::cout << i++ << "\t" << c << std::endl;
    }
    return false;
  }
  if (vm.count("config_id")) {
    std::ifstream file(resolveFile("cfg/configs.sph").string());
    std::vector<std::string> Configs;
    std::copy(std::istream_iterator<std::string>(file), std::istream_iterator<std::string>(),
              std::back_inserter(Configs));
    int32_t i = vm["config_id"].as<int>();
    if (i >= (int32_t)Configs.size() || i < 0) {
      std::cerr << "Not a valid config id" << std::endl;
      return false;
    }
    get<parameters::internal::config_file>() = resolveFile(Configs[i], config_folders).string();
  }
  if (vm.count("option")) {
    std::string config = get<parameters::internal::config_file>();

    std::stringstream ss;
    std::ifstream file(config);
    ss << file.rdbuf();
    file.close();

    boost::property_tree::ptree pt;
    boost::property_tree::read_json(ss, pt);

    auto options = pt.get_child_optional("options");
    if (options) {
      int32_t idx = 0;
      for (auto &child : options.get()) {
        if (idx == vm["option"].as<int>()) {
          for (auto &params : child.second) {
            for (auto &param : params.second) {
              std::stringstream option;
              option << params.first << "." << param.first << "=" << param.second.get_value<std::string>();
              std::cout << option.str() << std::endl;
              jsons.push_back(option.str());
            }
          }
        }
        idx++;
      }
    }
  }

  return true;
}
void arguments::cmd::enqueueRenderData(const std::map<std::string, std::vector<std::pair<float, float>>> &data) {
  for (const auto &[var, val] : data)
    renderData[var].push_back(val);
}

void arguments::cmd::finalize() {
  if (timers) {
    struct TimerData {
      std::string name;
      float average, median, min, max, dev;
      double total;
    };

    std::vector<TimerData> t_data;

    size_t max_len = 0;
    double frame_time = 0.0;

    for (auto t : TimerManager::getTimers()) {
      double total_time = 0.0;
      for (const auto &sample : t->getSamples())
        total_time += sample;
      auto stats = t->getStats();

      TimerData current{t->getDecriptor(), 
          stats ? stats.value().avg : 0.f, 
          stats ? stats.value().median : 0.f,
          stats ? stats.value().min : 0.f,
          stats ? stats.value().max : 0.f,
          stats ? stats.value().stddev : 0.f,  total_time};
      max_len = std::max(max_len, current.name.size());
      t_data.push_back(current);
      if (t->getDecriptor() == "Frame")
        frame_time = total_time;
    }

    int32_t number_field = 6;
    std::cout << std::endl;
    std::cout << std::setw(max_len + 3) << "Name"
              << " " << std::setw(number_field + 2) << "avg(ms)" << std::setw(number_field + 2) << "med(ms)"
              << std::setw(number_field + 2) << "min(ms)" << std::setw(number_field + 2) << "max(ms)"
              << std::setw(number_field + 2) << "dev(ms)" << std::setw(8) << "%"
              << "\t Total time" << std::endl;

    for (const auto &t : t_data) {
      std::cout << std::setw(max_len + 3) << t.name;
      std::cout << " ";
      std::cout << formatFloat(t.average);
      std::cout << formatFloat(t.median);
      std::cout << formatFloat(t.min);
      std::cout << formatFloat(t.max);
      std::cout << formatFloat(t.dev);
      std::cout << formatFloat(((t.total / frame_time) * 100.f));
      std::chrono::duration<double, std::milli> dur(t.total);
      auto seconds = std::chrono::duration_cast<std::chrono::seconds>(dur);
      auto minutes = std::chrono::duration_cast<std::chrono::minutes>(dur);
      auto remainder = t.total - seconds.count() * 1000.0;
      auto ms = static_cast<int32_t>(remainder);
      auto s = seconds.count() - minutes.count() * 60;
      std::cout << "\t";
      std::cout << minutes.count() << ":" << std::setfill('0') << std::setw(2) << s << ":" << std::setfill('0')
                << std::setw(3) << ms << std::setfill(' ') << std::endl;
      // std::cout << boost::format("%i:%02i.%03i") % minutes.count() % s % ms;

      // std::cout << std::endl;
    }
    if (renderData.size() != 0) {
      std::cout << "\n\nRender statistics:\n";
      int32_t len = renderData.begin()->second.size();
      std::cout << std::endl;
      int32_t framesInData = len;
      auto conv = [](auto v) { return pad(IO::config::convertToString(v)); };

      for (auto [tag, vals] : renderData) {
        // individual counters as tag -> vector of data
        if (std::any_of(vals.begin(), vals.end(), [](auto val) { return val.size() > 1; })) {
          std::cout << std::setw(12) << std::right << tag << ":\n";
          auto maxv = std::accumulate(vals.begin(), vals.end(), (std::size_t)0,
                                      [](auto acc, auto val) { return std::max(acc, val.size()); });
          std::vector<float> avgs(maxv), mins(maxv), maxs(maxv), devs(maxv);
          for (int32_t j = 0; j < maxv; ++j) {
            float avg = 0.f, dev = 0.f, min = FLT_MAX, max = -FLT_MAX;
            for (int32_t i = 0; i < framesInData; ++i) {
              min = std::min(min, j < vals[i].size() ? vals[i][j].first : FLT_MAX);
              max = std::max(max, j < vals[i].size() ? vals[i][j].first : -FLT_MAX);
              avg += j < vals[i].size() ? vals[i][j].first : 0.f;
            }
            avg /= (float)framesInData;
            for (int32_t i = 0; i < framesInData; ++i)
              dev += j < vals[i].size() ? powf(vals[i][j].first - avg, 2.f) : powf(avg, 2.f);
            dev /= (float)framesInData;
            avgs[j] = avg;
            devs[j] = dev;
            mins[j] = min;
            maxs[j] = max;
          }
          std::cout << std::setw(12) << std::right << "avg: ";
          for (auto a : avgs)
            std::cout << std::setw(8) << conv(a) << " ";
          std::cout << std::endl;
          std::cout << std::setw(12) << std::right << "min: ";
          for (auto a : mins)
            std::cout << std::setw(8) << conv(a) << " ";
          std::cout << std::endl;
          std::cout << std::setw(12) << std::right << "max: ";
          for (auto a : maxs)
            std::cout << std::setw(8) << conv(a) << " ";
          std::cout << std::endl;
          std::cout << std::setw(12) << std::right << "dev: ";
          for (auto a : devs)
            std::cout << std::setw(8) << conv(a) << " ";
          std::cout << std::endl;
          // sub frame information
        }
      }
      for (auto [tag, vals] : renderData) {
        if (std::any_of(vals.begin(), vals.end(), [](auto val) { return val.size() > 1; })) {
          continue;
        } else {
          std::cout << std::setw(12) << std::right << tag << " ";
          float avg = 0.f, dev = 0.f, min = FLT_MAX, max = -FLT_MAX;
          for (int32_t i = 0; i < framesInData; ++i) {
            min = std::min(min, vals[i][0].first);
            max = std::max(max, vals[i][0].first);
            avg += vals[i][0].first;
          }
          avg /= (float)framesInData;
          for (int32_t i = 0; i < framesInData; ++i)
            dev += powf(vals[i][0].first - avg, 2.f);
          dev /= (float)framesInData;
          std::cout << std::setw(8) << " avg: " << conv(avg) << std::setw(8) << " min: " << conv(min) << std::setw(8)
                    << " max: " << conv(max) << std::setw(8) << " dev: " << conv(dev);
          std::cout << std::endl;
        }
      }
      std::size_t frames = 0, subSteps = 0;
      for (auto [tag, vals] : renderData) {
        frames = std::max(frames, vals.size());
        for (int32_t s = 0; s < vals.size(); ++s)
          subSteps = std::max(subSteps, vals[s].size());
      }
      //if (frames != 0) {
      //  std::cout << "\n\n\nRaw Render Data for " << frames << " frames with " << subSteps << " substeps."
      //            << std::endl;
      //  std::cout << "Data ordering:\n";
      //    for (auto [tag, vals] : renderData) {
      //      // std::cout << tag << std::endl;
      //      if (boost::any_of(vals.begin(), vals.end(), [](auto val) { return val.size() > 1; })) {
      //        for (int32_t s = 0; s < subSteps; ++s) {
      //          if (s == 0)
      //            std::cout << std::setw(20) << std::right << tag;
      //          else
      //            std::cout << std::setw(20) << " ";
      //        }
      //      } else
      //        std::cout << std::setw(20) << std::right << tag;
      //      std::cout << " | ";
      //  }
      //  std::cout << std::endl;
      //  for (int32_t i = 0; i < frames; ++i) {
      //    for (auto [tag, vals] : renderData) {
      //      //std::cout << tag << std::endl;
      //      if (boost::any_of(vals.begin(), vals.end(), [](auto val) { return val.size() > 1; })) {
      //        for (int32_t s = 0; s < subSteps; ++s) {
      //          if (s >= vals.size())
      //            std::cout << conv(0.f) << "@" << conv(0.f) << " ";
      //          else
      //            std::cout << conv(vals[i][s].first) << "@" << conv(vals[i][s].second) << " ";
      //        }
      //      } else
      //        std::cout << conv(vals[i][0].first) << "@" << conv(vals[i][0].second) << " ";
      //      std::cout << " | ";
      //    }
      //    std::cout << "\n";
      //  }
      //}
    }
  }
  if (vm.count("config_log")) {
    IO::config::save_config(vm["config_log"].as<std::string>());
  }
  if (vm.count("log")) {
    logger::write_log(vm["log"].as<std::string>());
  }
  if (vm.count("memory") || vm.count("info")) {
    struct MemoryInfo {
      std::string name, type_name;
      size_t allocSize, elements;
      bool swap;
    };

    std::vector<MemoryInfo> mem_info;
    size_t longest_name = 0;
    size_t longes_type = 0;

    for_each(arrays_list, [&](auto x) {
      using T = decltype(x);
      using type = typename T::type;
      auto allocSize = T::alloc_size;
      auto elements = allocSize / sizeof(type);
      std::string name = T::variableName;
      std::string t_name = type_name<type>();
      bool swap = false;

      if constexpr (has_rear_ptr<T>)
        swap = true;
      MemoryInfo current{name, t_name, allocSize, elements, swap};
      if (allocSize != 0) {
        longest_name = std::max(longest_name, current.name.size());
        longes_type = std::max(longes_type, current.type_name.size());
        mem_info.push_back(current);
      }
    });
    if (vm.count("memory")) {
      std::cout << std::endl;
      std::cout << std::setw(longest_name + 3) << "Name"
                << " " << std::setw(longes_type + 3) << "Type"
                << " " << std::setw(8) << "size" << std::endl;
    }
    size_t total = 0;
    size_t total_fixed = 0;
    size_t total_dynamic = 0;
    std::sort(mem_info.begin(), mem_info.end(),
              [](const auto &lhs, const auto &rhs) { return lhs.allocSize < rhs.allocSize; });
    for (const auto &t : mem_info) {
      if (vm.count("memory")) {
        std::cout << std::setw(longest_name + 3) << t.name;
        std::cout << " ";
        std::cout << std::setw(longes_type + 3) << t.type_name;
        std::cout << " ";
        if (t.swap)
          std::cout << std::setw(8) << IO::config::bytesToString(2 * t.allocSize);
        else
          std::cout << std::setw(8) << IO::config::bytesToString(t.allocSize);
        std::cout << std::endl;
      }
      if (t.swap)
        total += 2 * t.allocSize;
      else
        total += t.allocSize;
      if (t.swap)
        total_fixed += t.allocSize;
    }
    for (auto &alloc : MemoryManager::instance().allocations)
      total_dynamic += alloc.allocation_size;

    std::cout << "Total memory consumption: " << IO::config::bytesToString(total) << std::endl;
    std::cout << "const memory consumption: " << IO::config::bytesToString(total_fixed) << std::endl;
    std::cout << "dyn   memory consumption: " << IO::config::bytesToString(total_dynamic) << std::endl;
  }
  if (vm.count("info")) {
    auto end = std::chrono::high_resolution_clock::now();
    auto dur = end - start;

    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(dur);
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(dur);
    auto minutes = std::chrono::duration_cast<std::chrono::minutes>(dur);
    auto ms = milliseconds.count() - seconds.count() * 1000;
    auto s = seconds.count() - minutes.count() * 60;
    std::cout << "Total time:               ";
    // std::cout << boost::format("%i:%02i.%03i") % minutes.count() % s % ms;
    std::cout << minutes.count() << ":" << std::setfill('0') << std::setw(2) << s << ":" << std::setw(3)
              << std::setfill('0') << ms;
    std::cout << std::endl;
    std::cout << "Number of particles:      " << IO::config::numberToString(get<parameters::internal::num_ptcls>()) << " of "
              << IO::config::numberToString(get<parameters::simulation_settings::maxNumptcls>()) << std::endl;
    std::cout << "Frames:                   " << get<parameters::internal::frame>() << std::endl;
    std::cout << "Simulated time:           " << get<parameters::internal::simulationTime>() << "s" << std::endl;
  }

  if (vm.count("neighbordump")) {
    int32_t *neighbor_list_shared = get<arrays::neighborList>();
    int32_t alloc_size = (int32_t)info<arrays::neighborList>().alloc_size;
    void *neighbor_list_local = malloc(alloc_size);

    int32_t *offsetList_shared = get<arrays::neighborListLength>();
    int32_t alloc_size_offsetList = (int32_t)info<arrays::neighborListLength>().alloc_size;
    void *offsetList_local = malloc(alloc_size_offsetList);

    cuda::memcpy(neighbor_list_local, neighbor_list_shared, alloc_size, cudaMemcpyDeviceToHost);
    cuda::memcpy(offsetList_local, offsetList_shared, alloc_size_offsetList, cudaMemcpyDeviceToHost);

    std::string fileName = vm["neighbordump"].as<std::string>();

    std::ofstream ofs(fileName, std::ios::binary);
    int32_t max_num = get<parameters::simulation_settings::maxNumptcls>();
    int32_t num = get<parameters::internal::num_ptcls>();
    ofs.write(reinterpret_cast<char *>(&max_num), sizeof(int32_t));
    ofs.write(reinterpret_cast<char *>(&num), sizeof(int32_t));
    ofs.write(reinterpret_cast<char *>(&alloc_size), sizeof(int32_t));
    ofs.write(reinterpret_cast<char *>(&alloc_size_offsetList), sizeof(int32_t));
    ofs.write(reinterpret_cast<char *>(neighbor_list_local), alloc_size);
    ofs.write(reinterpret_cast<char *>(offsetList_local), alloc_size_offsetList);
    ofs.close();
  }
  if (vm.count("params")) {
    size_t max_len = 0;
    for (auto p : param_watchlist) {
      max_len = std::max(max_len, p->name().size());
    }

    int32_t number_field = 8;
    std::cout << std::endl;
    std::cout << std::setw(max_len + 3) << "Name"
              << "    " << std::setw(number_field + 2) << "avg    " << std::setw(number_field + 2) << "med    "
              << std::setw(number_field + 2) << "min    " << std::setw(number_field + 2) << "max    "
              << std::setw(number_field + 2) << "dev    " << std::endl;

    for (const auto &p : param_watchlist) {
      std::cout << p->to_string(max_len + 3) << std::endl;
    }
  }
  printLoggedParameters();

  auto &mem = MemoryManager::instance();
  for (auto &allocation : mem.allocations) {
    if (allocation.ptr == nullptr)
      continue;

    for_each(arrays_list, [allocation](auto x) {
      using T = decltype(x);
      if (T::ptr == allocation.ptr)
        T::ptr = nullptr;
    });
    cudaFree(allocation.ptr);
  }
  // Graceful shutdown disabled due to issues on current linux versions.
  for_each(arrays_list, [](auto x) {
    using T = decltype(x);
    if (T::valid() && T::ptr != nullptr)
      T::free();
  });
}

void arguments::cmd::parameter_stats() {
  if (vm.count("params")) {
    for (auto &p : param_watchlist)
      p->sample();
  }
}

boost::program_options::variables_map &arguments::cmd::getVM() { return vm; }
