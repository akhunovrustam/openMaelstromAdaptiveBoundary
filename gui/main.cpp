#include <tools/exceptionHandling.h>
#include "glui.h"
#include <sstream>
#include <thread>
#include <utility/helpers/arguments.h>
#include <simulation/particleSystem.h> 
#include <simulation/functions.h>
#ifdef WIN32
#include <windows.h>
#endif
#include <iostream> 

#include <chrono>
enum struct context { cpu, gpu, both };

template<typename T>
using deleted_unique_ptr = std::unique_ptr<T[], std::function<void(T*)>>;

struct ArrayProperties {
    const std::size_t elementSize;
    const std::size_t defaultElements = 1;
    const std::type_index typeInfo;
    const std::vector<std::type_index> alternativeTypes;
    const std::function<bool(void)> validFn = []() {return true; };
    const std::string mangledName;
};

struct Array {
    void* d_ptr = nullptr, *h_ptr = nullptr;
    const ArrayProperties props;
    const std::string identifier;
    const std::string nameSpace;
    std::size_t currentAllocSize = 0;
    std::chrono::system_clock::time_point 
        lastAllocTime{ std::chrono::milliseconds{ 0 } }, 
        lastAccessTime{ std::chrono::milliseconds{ 0 } },
        lastFreeTime{ std::chrono::milliseconds{ 0 } },
        lastTempTime{ std::chrono::milliseconds{ 0 } };

    friend std::ostream& operator << (std::ostream& out, const Array& c);

    template<typename T = void>
    T* allocate(context c = context::gpu, std::size_t elements = 0) {
        if (!std::is_same_v<T, void>) {
            if (typeid(std::remove_cv_t<T>) != props.typeInfo && std::find(props.alternativeTypes.begin(), props.alternativeTypes.end(), typeid(std::remove_cv_t<T>)) == props.alternativeTypes.end()) {
                std::cerr << "Error: Allocate called with invalid type " << typeid(T).name() << std::endl;
                throw std::invalid_argument("Allocated called with invalid type.");
            }
        }
        if (c == context::both) {
            std::cerr << "Error: allocate does not support context::both\n";
            throw std::invalid_argument("Error: allocate does not support context::both");
        }
        if (elements == 0 && currentAllocSize == 0) elements = props.defaultElements;
        else if(elements == 0) elements = currentAllocSize / props.elementSize;
        if (c == context::gpu) {
            if (d_ptr != nullptr) {
                std::clog << "Warning: Array " << nameSpace << "." << identifier << " already allocated on " << (c == context::gpu ? "gpu" : "cpu" ) << " with " << currentAllocSize << "B. Allocating again with " << elements * props.elementSize << "B." << std::endl;
                void* temporary = nullptr;
                std::clog << "Reallocating device data" << std::endl;
                cudaMalloc(&temporary, elements * props.elementSize);
                cudaMemcpy(temporary, d_ptr, currentAllocSize, cudaMemcpyDeviceToDevice);
                cudaFree(d_ptr);
                d_ptr = temporary;
            }
            else if (h_ptr != nullptr) {
                std::clog << "Array " << nameSpace << "." << identifier << " already allocated on " << (c != context::gpu ? "cpu" : "gpu") << "." << std::endl;
                if (elements * props.elementSize != currentAllocSize) {
                    std::clog << "Error: gpu and cpu arrays are not of equal size." << std::endl;
                    throw std::invalid_argument("GPU and CPU Array dimensions mismatched");
                }
                void* temporary = nullptr;
                cudaMalloc(&d_ptr, elements * props.elementSize);
                std::clog << "Copying device data from host" << std::endl;
                cudaMemcpy(d_ptr, h_ptr, currentAllocSize, cudaMemcpyHostToDevice);
            }
            else {
                cudaMalloc(&d_ptr, elements * props.elementSize);
            }
        }
        else if (c == context::cpu) {
            if (h_ptr != nullptr) {
                std::clog << "Warning: Array " << nameSpace << "." << identifier << " already allocated on " << (c != context::gpu ? "gpu" : "cpu") << " with " << currentAllocSize << "B. Allocating again with " << elements * props.elementSize << "B." << std::endl;
                std::clog << "Reallocating host data" << std::endl;
                h_ptr = realloc(h_ptr, elements * props.elementSize);
            }
            else if (d_ptr != nullptr) {
                std::clog << "Array " << nameSpace << "." << identifier << " already allocated on " << (c == context::gpu ? "gpu" : "cpu") << "." << std::endl;
                if (elements * props.elementSize != currentAllocSize) {
                    std::cerr << "Error: gpu and cpu arrays are not of equal size." << std::endl;
                    throw std::invalid_argument("GPU and CPU Array dimensions mismatched");
                }
                h_ptr = malloc(elements * props.elementSize);
                std::clog << "Copying device data to host" << std::endl;
                cudaMemcpy(d_ptr, h_ptr, currentAllocSize, cudaMemcpyDeviceToHost);
            }
            else {
                h_ptr = malloc(elements * props.elementSize);
            }
        }
        currentAllocSize = elements * props.elementSize;
        lastAllocTime = std::chrono::system_clock::now();
        return (T*)(c == context::cpu ? h_ptr : d_ptr);
    }
    void free(context c = context::gpu, bool copyBack = true) {
        if (c == context::gpu) {
            if (copyBack && h_ptr != nullptr) {
                cudaMemcpy(h_ptr, d_ptr, currentAllocSize, cudaMemcpyDeviceToHost);
            }
            cudaFree(d_ptr);
            d_ptr = nullptr;
        }
        else if (c == context::cpu) {
            if (copyBack && d_ptr != nullptr) {
                cudaMemcpy(d_ptr, h_ptr, currentAllocSize, cudaMemcpyHostToDevice);
            }
            ::free(h_ptr);
            h_ptr = nullptr;
        }
        else {
            cudaFree(d_ptr);
            ::free(h_ptr);
            d_ptr = nullptr;
            h_ptr = nullptr;
        }
        if (d_ptr == nullptr && h_ptr == nullptr) currentAllocSize = 0;
        lastFreeTime = std::chrono::system_clock::now();
    }
    void copy(void* target, context tD = context::gpu, context tS = context::both, std::size_t elements = 0) {
        if (tD == context::both) {
            std::cerr << "Error: allocate does not support target context::both\n";
            throw std::invalid_argument("Error: allocate does not target support context::both");
        }
        if ((tS == context::cpu && d_ptr == nullptr) || (tS == context::cpu && h_ptr == nullptr)) {
            std::cerr << "Trying to copy from unallocated context::" << (tS == context::gpu ? "gpu" : (tS == context::both ? "both" : "cpu")) << std::endl;
            throw std::invalid_argument("Trying to copy from unallocated context.");
        }
        auto dir = cudaMemcpyDeviceToDevice;
        if ((tD == context::gpu && tS == context::gpu) || 
            (tD == context::gpu && (tS == context::both && d_ptr != nullptr)))
            dir = cudaMemcpyDeviceToDevice;
        if ((tD == context::cpu && tS == context::cpu) ||
            (tD == context::cpu && (tS == context::both && h_ptr != nullptr)))
            dir = cudaMemcpyHostToHost;
        if ((tD == context::cpu && tS == context::gpu) ||
            (tD == context::cpu && (tS == context::both && h_ptr == nullptr)))
            dir = cudaMemcpyDeviceToHost;
        if ((tD == context::gpu && tS == context::cpu) ||
            (tD == context::gpu && (tS == context::both && d_ptr == nullptr)))
            dir = cudaMemcpyHostToDevice;
        if (dir != cudaMemcpyHostToHost) 
            cudaMemcpy(target, (dir == cudaMemcpyDeviceToDevice && dir == cudaMemcpyDeviceToHost ? d_ptr : h_ptr), elements == 0 ? currentAllocSize : elements * props.elementSize, dir);
        else
            memcpy(target, h_ptr, elements == 0 ? currentAllocSize : elements * props.elementSize);
        lastAccessTime = std::chrono::system_clock::now();
    }

    template<typename T>
    deleted_unique_ptr<T> getTemporary(context c = context::gpu, std::size_t elements = 0) {
        if (c == context::both) {
            std::cerr << "Error: allocate does not support target context::both\n";
            throw std::invalid_argument("Error: allocate does not target support context::both");
        }
        if (elements == 0 && currentAllocSize == 0) elements = props.defaultElements;
        else if (elements == 0) elements = currentAllocSize / props.elementSize;
        else if (elements != 0 && elements != currentAllocSize) {
            std::cerr << "Error: requested temporary memory size " << elements * props.elementSize << "B does not match currently allocated size " << currentAllocSize << "B.\n";
            throw std::invalid_argument("Mismatched array dimensions");
        }
        lastTempTime = std::chrono::system_clock::now();
        void* tPtr = nullptr;
        if (c == context::gpu) {
            cudaMalloc(&tPtr, elements * props.elementSize);
            if (d_ptr != nullptr) {
                cudaMemcpy(tPtr, d_ptr, currentAllocSize, cudaMemcpyDeviceToDevice);
            }
            else if (h_ptr != nullptr) {
                cudaMemcpy(tPtr, h_ptr, currentAllocSize, cudaMemcpyHostToDevice);
            }
        }
        else{
            tPtr = malloc(elements * props.elementSize);
            if (d_ptr != nullptr) {
                cudaMemcpy(tPtr, d_ptr, currentAllocSize, cudaMemcpyDeviceToHost);
            }
            else if (h_ptr != nullptr) {
                memcpy(tPtr, h_ptr, currentAllocSize);
            }
        }

        return deleted_unique_ptr<T>((T*) tPtr, [=](T* ptr) {
            // std::clog << "Destroying temporary pointer \n";
            if (c == context::cpu) {
                if (h_ptr != nullptr)
                    memcpy(h_ptr, ptr, currentAllocSize);
                if (d_ptr != nullptr)
                    cudaMemcpy(d_ptr, ptr, currentAllocSize, cudaMemcpyHostToDevice);
                ::free(ptr);
            }
            if (c == context::gpu) {
                if (h_ptr != nullptr)
                    cudaMemcpy(h_ptr, ptr, currentAllocSize, cudaMemcpyDeviceToHost);
                if (d_ptr != nullptr)
                    cudaMemcpy(d_ptr, ptr, currentAllocSize, cudaMemcpyDeviceToDevice);
                cudaFree(ptr);
            }
            });
    }
    
    template<typename T = void>
    T* get(context c = context::both) {
        if (!std::is_same_v<T, void>) {
            if (typeid(std::remove_cv_t<T>) != props.typeInfo && std::find(props.alternativeTypes.begin(), props.alternativeTypes.end(), typeid(std::remove_cv_t<T>)) == props.alternativeTypes.end()) {
                std::cerr << "Error: Allocate called with invalid type " << typeid(T).name() << std::endl;
                throw std::invalid_argument("Allocated called with invalid type.");
            }
        }
        if(c == context::gpu && d_ptr == nullptr || c == context::cpu && h_ptr == nullptr || (c==context::both && h_ptr == nullptr && d_ptr == nullptr)){
            std::cerr << "Error: trying to get unallocated memory\n";
            throw std::invalid_argument("Error: trying to get unallocated memory");
        }
        lastAccessTime = std::chrono::system_clock::now();
        if (c == context::gpu) return (T*)d_ptr;
        else if (c == context::cpu) return (T*)h_ptr;
        else return (T*)(d_ptr == nullptr ? h_ptr : d_ptr);

    }
};
std::ostream& operator << (std::ostream& out, const Array& arr) {
    std::time_t lastAllocTime = std::chrono::system_clock::to_time_t(arr.lastAllocTime);
    std::time_t lastFreeTime = std::chrono::system_clock::to_time_t(arr.lastFreeTime);
    std::time_t lastAccessTime = std::chrono::system_clock::to_time_t(arr.lastAccessTime);
    std::time_t lastTempTime = std::chrono::system_clock::to_time_t(arr.lastTempTime);
    out <<
        "Array " << arr.nameSpace << "." << arr.identifier << ":\n" <<
        "Currently allocated: " << (arr.currentAllocSize != 0 ? "yes" : "no") << "\n" <<
        "Type: " << arr.props.mangledName << " [ element size " << arr.props.elementSize << " ]" << "\n" <<
        "Default elements: " << arr.props.defaultElements << "\n" <<
        "Valid: " << (arr.props.validFn() ? "yes" : "no") << "\n" <<
        "Type index: " << arr.props.typeInfo.hash_code() << "\n" <<
        "Device ptr: " << arr.d_ptr << "\n" <<
        "Host ptr: " << arr.h_ptr << "\n" <<
        "Last allocation time: " << std::put_time(std::localtime(&lastAllocTime), "%F %T") << "\n"<<
        "Last access time: " << std::put_time(std::localtime(&lastAccessTime), "%F %T") << "\n" <<
        "Last free time: " << std::put_time(std::localtime(&lastFreeTime), "%F %T") << "\n" <<
        "Last temporary time: " << std::put_time(std::localtime(&lastTempTime), "%F %T") << "\n" << std::endl;;
    return out;
}

class ArrayManager {
    ArrayManager() {}
    std::map<std::string, Array*> qidArrayMap;
    std::multimap<std::string, std::string> qidMap;

    std::string resolveArray(std::string s) {
        if (s.find(".") != std::string::npos) return s;
        auto [ns, id] = split(s);
        auto idc = qidMap.count(id);
        if (idc == 0)
            throw std::invalid_argument("Parameter " + id + " does not exist");
        if (idc > 1 && ns == "")
            throw std::invalid_argument("Parameter " + id + " is ambiguous");
        auto range = qidMap.equal_range(id);
        auto qIdentifier = range.first->second;
        if (idc > 1)
            for (auto i = range.first; i != range.second; ++i)
                if (ns == i->second.substr(0, i->second.find(".")))
                    qIdentifier = i->second;
        return qIdentifier;
    }
    bool ArrayExists(std::string s) {
        if (qidArrayMap.find(s) == qidArrayMap.end()) return false;
        return true;
    }

    ArrayManager(const ArrayManager&) = delete;
public:
    bool isAmbiguous(std::string ident){
        auto idc = qidMap.count(ident);
        if (idc == 0)
            return false;
        //throw std::invalid_argument("Parameter " + id + " does not exist");
        if (idc > 1)
            return true;
        //throw std::invalid_argument("Parameter " + id + " is ambiguous");
        return false;
    }

    static ArrayManager& instance() {
        static ArrayManager inst;
        return inst;
    }

    Array& getArray(std::string identifier) {
        auto qid = resolveArray(identifier);
        if (!ArrayExists(qid)) throw std::invalid_argument("Parameter " + identifier + " does not exist");
        return *qidArrayMap[qid];
    }
    Array& addNewArray(std::string identifier, ArrayProperties props) {
        if (qidArrayMap.find(identifier) != qidArrayMap.end())
            throw std::invalid_argument("Parameter " + identifier + " already exists.");
        auto nsid = split(identifier);
        auto ns = nsid.first;
        auto id = nsid.second;
        qidMap.insert(std::make_pair(id, identifier));
        qidArrayMap[identifier] = new Array{ .props = props, .identifier = id, .nameSpace = ns };
        return *qidArrayMap[identifier];
    }
};

template<typename T, typename... Ts>
ArrayProperties getProps(std::size_t defaultElems, std::function<bool(void)> validFn = []() {return true; }) {
    std::type_index id = typeid(T);
    std::size_t elemSize = sizeof(T);
    std::vector<std::type_index> altIds;
    std::tuple<Ts...> t;
    std::apply([&altIds](auto... v) {(altIds.push_back(typeid(v)), ...); }, t);
    return ArrayProperties{
        .elementSize = elemSize,
        .defaultElements = defaultElems,
        .typeInfo = id,
        .alternativeTypes = altIds,
        .validFn = validFn,
        .mangledName = typeid(T).name()
    };
}


#ifdef WIN32
BOOL CtrlHandler(DWORD fdwCtrlType){
    std::clog << "Caught signal " << fdwCtrlType << std::endl;
    switch (fdwCtrlType)
    {
    case CTRL_CLOSE_EVENT:
        GUI::instance().quit();
        GUI::instance().render_lock.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        return(TRUE);

    default:
        return FALSE;
    }
}
int main(int argc, char *argv[]) try {
    #else
int main(int argc, char *argv[]) {
    #endif
    //auto& aman = ArrayManager::instance();
    //int32_t n = 32;
    //aman.addNewArray("test.array", getProps<float, uFloat<SI::m>>(n)).allocate(context::cpu);
    //auto& testArray = aman.getArray("array");
    //std::cout << testArray << std::endl;
    //auto ptr = testArray.get<float>();
    //std::cout << "Ptr: " << ptr << std::endl;
    //for (int32_t i = 0; i < n; ++i) {
    //    ptr[i] = (float)i;
    //}
    //testArray.allocate(context::gpu);
    //testArray.free(context::cpu);

    //{
    //    auto temp = testArray.getTemporary<float>(context::cpu);
    //    for (int32_t i = 0; i < n; ++i) {
    //        std::cout << temp[i] << " ";
    //        temp[i] = (float)(n - i);
    //    }
    //}

    //testArray.allocate(context::cpu);
    //testArray.free(context::gpu);
    //auto ptr2 = testArray.get<float>();
    //for (int32_t i = 0; i < n; ++i) {
    //    std::cout << ptr2[i] << " ";
    //}
    //std::cout << std::endl;

    //return 1;
    

#ifdef WIN32
  if (!SetConsoleCtrlHandler((PHANDLER_ROUTINE)CtrlHandler, TRUE)) {
    std::cerr << "Could not set CtrlHandler. Exiting." << std::endl;
    return 0;
  }
  _set_se_translator(translator);
#endif 
  auto &gui = GUI::instance();
  gui.render_lock.lock();
  gui.initParameters(argc, argv);
  gui.initGVDB();
  gui.initSimulation();

  #ifdef WIN32
  SPH::logDump::create_log_folder();
  std::thread render_thread([&]() {
    try {
      gui.initGL();
      gui.renderLoop();
    }
    CATCH_DEFAULT
  });
  auto &cmd_line = arguments::cmd::instance();
  try {
    while (!gui.render_lock.try_lock()) {
      cuda_particleSystem::instance().step();
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      if ((cmd_line.end_simulation_frame && get<parameters::internal::frame>() >= cmd_line.timesteps) ||
          (cmd_line.end_simulation_time && get<parameters::internal::simulationTime>() >= cmd_line.time_limit))
        break;
      if (cmd_line.pause_simulation_time && get<parameters::internal::simulationTime>() >= cmd_line.time_pause) {
        cuda_particleSystem::instance().running = false;
        arguments::cmd::instance().pause_simulation_time = false;
        cmd_line.time_pause = FLT_MAX;
      }
    }
    gui.shouldStop = true;
  }
  CATCH_FN(
      gui.quit();
      )

  #else
  std::thread render_thread([&]() {
    //try {
      gui.initGL();
      gui.renderLoop();
    //}
    //CATCH_DEFAULT
  });
  auto &cmd_line = arguments::cmd::instance();
  //try {
    while (!gui.render_lock.try_lock()) {
      cuda_particleSystem::instance().step();
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      if ((cmd_line.end_simulation_frame && get<parameters::internal::frame>() >= cmd_line.timesteps) ||
          (cmd_line.end_simulation_time && get<parameters::internal::simulationTime>() >= cmd_line.time_limit))
        break;
      if (cmd_line.pause_simulation_time && get<parameters::internal::simulationTime>() >= cmd_line.time_pause) {
        cuda_particleSystem::instance().running = false;
        arguments::cmd::instance().pause_simulation_time = false;
        cmd_line.time_pause = FLT_MAX;
      }
    }
    gui.shouldStop = true;
  //}
  //CATCH_FN(
      gui.quit();
 //     )

  #endif
  render_thread.join();
  cmd_line.finalize();
  return 0;
}
#ifdef WIN32
CATCH_DEFAULT
#endif