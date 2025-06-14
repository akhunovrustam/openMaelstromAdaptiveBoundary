#include <cuda_runtime.h>
#include <simulation/particleSystem.h>
#include <utility/include_all.h>
#include <utility/helpers/arguments.h>
#include <config/config.h>
#include <boost/filesystem.hpp>  
namespace fs = boost::filesystem;

int main(int argc, char *argv[]) {
  auto binary_directory = fs::system_complete(fs::path(argv[0])).parent_path();
  auto working_directory = fs::current_path().parent_path();

  get<parameters::internal::working_directory>() = working_directory.string();
  get<parameters::internal::binary_directory>() = binary_directory.string();
  get<parameters::internal::source_directory>() = sourceDirectory;
  get<parameters::internal::build_directory>() = binaryDirectory;

  cudaSetDevice(0);
  auto &cmd_line = arguments::cmd::instance();
  if (!cmd_line.init(true, argc, argv))
    return 0;

  cuda_particleSystem::instance().init_simulation();
  cuda_particleSystem::instance().running = false;
  cuda_particleSystem::instance().step();
  cuda_particleSystem::instance().running = true;

  try {
    if (cmd_line.end_simulation_frame) {
      while (get<parameters::internal::frame>() < cmd_line.timesteps) {
		  cuda_particleSystem::instance().renderFlag = true;
        cuda_particleSystem::instance().step();
        arguments::loadbar(get<parameters::internal::frame>(), cmd_line.timesteps, 80);
      } 
    } else if(cmd_line.end_simulation_time){
      while (get<parameters::internal::simulationTime>() < cmd_line.time_limit) {
		  cuda_particleSystem::instance().renderFlag = true;
        cuda_particleSystem::instance().step();

        arguments::loadbar((int32_t)(get<parameters::internal::simulationTime>() / cmd_line.time_limit * 100.f), 100, 80);
      }
    }else{{
      while (get<parameters::internal::simulationTime>() < 1.0f) {
		  cuda_particleSystem::instance().renderFlag = true;
        cuda_particleSystem::instance().step();

        arguments::loadbar((int32_t)(get<parameters::internal::simulationTime>() / 1.f * 100.f), 100, 80);
      }
    }}

  } catch (std::exception e) {
    std::cout << "Caught exception while running simulation: " << e.what() << std::endl;
  }

  cmd_line.finalize();
     
  return 0;
}
 