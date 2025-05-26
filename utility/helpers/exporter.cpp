#include <utility/helpers/exporter.h>

void dumpParamsBar(float progress) {
	int barWidth = 70;
	std::cout << "Writing parameters ";
	std::cout << "[";
	int pos = (int32_t)((float)barWidth * progress);
	for (int i = 0; i < barWidth; ++i) {
		if (i < pos) std::cout << "=";
		else if (i == pos) std::cout << ">";
		else std::cout << " ";
	}
	std::cout << "] " << std::setw(3) << int(progress * 100.0) << " %\r";
	std::cout.flush();
}
void dumpArraysBar(float progress) {
	int barWidth = 70;
	std::cout << "Writing Arrays     ";
	std::cout << "[";
	int pos = (int32_t)((float)barWidth * progress);
	for (int i = 0; i < barWidth; ++i) {
		if (i < pos) std::cout << "=";
		else if (i == pos) std::cout << ">";
		else std::cout << " ";
	}
	std::cout << "] " << std::setw(3) << int(progress * 100.0) << " %\r";
	std::cout.flush();
}



template<typename... Ts>
auto writeParameters(std::tuple<Ts...>, std::ofstream& dump_file) {
	(IO::writeParameter<Ts>(dump_file), ...);
}

template<typename... Ts>
auto writeArrays(std::tuple<Ts...>, std::ofstream& dump_file) {
	(IO::writeArray<Ts>(dump_file), ...);
}
#include <yaml-cpp/yaml.h>
void IO::dumpAll(std::string filename) {
	//std::cerr << ("Dump file not currently supported") << std::endl;
  std::ofstream dump_file;
  dump_file.open(filename, std::ios::binary);

  //writeParameters(uniforms_list, dump_file);
  writeArrays(arrays_list, dump_file);
  //YAML::hexFloatConversion = true;
  auto tree = ParameterManager::instance().buildTree();
  //YAML::hexFloatConversion = false;
  YAML::Emitter out;
  out << tree;
  auto fName = filename.substr(0, filename.find_last_of("."));
  fName = fName + ".yaml";
  std::ofstream config_file;
  config_file.open(fName);
  config_file << out.c_str();
  std::cout << "Done writing dump file " << filename;
}