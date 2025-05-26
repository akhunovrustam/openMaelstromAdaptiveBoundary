#include <SPH/logDump/logDump.cuh>
#include <utility/include_all.h>
#include <stdlib.h>
#include <fstream>
#include <filesystem>
#include <boost/filesystem.hpp>

namespace fs = std::experimental::filesystem;

#define range 100
#define fullrange (range*2+1) 
#define deviation_factor (range / 2.5f)
#define logfactor 10

std::string fulldump = "statistics.ssv";
std::string rigiddump = "rigid_dump";

const std::string currentDateTime() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
   
    
    strftime(buf, sizeof(buf), "%Y-%m-%d.%H_%M_%S", &tstruct);

    return buf;
}

struct stats{
    float min, max, avg, short_avg, xt, deviation, median;
    std::array<float, fullrange> val_bins;
    std::array<float, fullrange> val_bins_i;

    stats()
    {
        val_bins.fill(0.f);
        val_bins_i.fill(0);
    }

    int tmp = 1;
    std::string to_string()
    {
        std::string ret = std::to_string(max) + " " + std::to_string(min) + " " + std::to_string(avg) + " " + std::to_string(short_avg)
        + " " + std::to_string(xt) + " " + std::to_string(deviation) + " " + std::to_string(median) + "\t";

        // for (int i = 0; i < val_bins.size(); i++)
        //     ret += " " + std::to_string(val_bins[i]);
        
        // ret += "\t";

        for (int i = 0; i < val_bins_i.size(); i++)
            ret += (i != 0 ? " " : "") + std::to_string(val_bins_i[i]);

        return ret;
    }
};

stats calculateStats(std::vector<float> vals, float distmax = 20.f){
    stats retstat;
    float statThreshold = 100.f;
    float lgfactor = distmax / fullrange;

    int fluid_parts = get<parameters::internal::num_ptcls_fluid>();
    int statNumber = ceil(fluid_parts * statThreshold / 100.f);

    std::sort(vals.begin(), vals.end(), std::less<float>());
    
        
    std::vector<float> forstat(vals.begin(), vals.begin() + statNumber);
    
    // for (int i = 0; i < 40; i++)
    //     std::cout << forstat[i] << std::endl;
    
    retstat.avg = vals.back() / fluid_parts;
    int cntri = (int)floor(statNumber/2.f);
    retstat.median = forstat[cntri];
    vals.pop_back();
    retstat.min = vals.front();
    retstat.max = vals.back();
    retstat.xt = forstat.back();
    
    float dev_sum = 0.f;
    float short_sum = 0.f;
    for (auto v:forstat)
        short_sum += v;
  
    retstat.short_avg = short_sum/statNumber;

    for (auto v:forstat)
        dev_sum += (v - retstat.short_avg)*(v - retstat.short_avg);
  
    retstat.deviation = sqrt(dev_sum/statNumber);

    int i = 0;
    int rightc = 0, leftc = 0, centerc = 0;
    int rightcv = 0, leftcv = 0, centercv = 0;
    for (auto v:forstat)
    {

        int index = ceil(((v) - retstat.median) / (retstat.deviation/deviation_factor) ) + range;
        if (index < 30) leftc++;
        if (index > 30) rightc++;
        if (index == 30) centerc++;

        if (v < retstat.median) leftcv++;
        if (v > retstat.median) rightcv++;
        if (v == retstat.median) centercv++;

        if (index < 30 && v > retstat.median)
        {
            std::cout << "bug " << v << " " << retstat.median << " " << index << std::endl;
            break;
        }
        // if (i++ > 30) break;
    }
    int mindex = ceil(((retstat.median) - retstat.median) / (retstat.deviation/deviation_factor) ) + range;

    // std::cout << "indexv " << leftcv << " " << rightcv << " " << centercv << std::endl;
    // std::cout << "index " << leftc << " " << rightc << " " << centerc << " stat " << statNumber << " " << cntri << " " << mindex
    // << " " << retstat.min << " " << retstat.xt << " " << retstat.median << std::endl;

    for (auto v:forstat)
    {
        if (v < 0.f) v = 0.f;
        // int index = ceil(((v + retstat.deviation/(deviation_factor*2.f)) - retstat.median) / (retstat.deviation/deviation_factor) ) + range;
        // if (index < 0) index = 0;
        // if (index >= fullrange) index = fullrange - 1;
        int index = 0;
        if (v > fullrange * lgfactor) index = fullrange - 1; 
        else index = floor(v / lgfactor);
        
        if (distmax > 41.f && v > 0.f && retstat.max == v) std::cout << index << " " << v << " " << lgfactor 
            << " " << (fullrange * lgfactor) << " " << fullrange << " " << (v > fullrange * lgfactor) << std::endl;
        
        if (index > fullrange - 1) index = fullrange - 1; 

        // std::cout << "index = " << index << std::endl;
        retstat.val_bins[index] += v;
        retstat.val_bins_i[index] += 1;
    }

    float total = 0.f;
    for (int i = 0; i < retstat.val_bins_i.size(); i++)
    {
        retstat.val_bins[i] = retstat.val_bins_i[i] == 0.f ? 0.f : retstat.val_bins[i] / retstat.val_bins_i[i];
        retstat.val_bins_i[i] = retstat.val_bins_i[i] / statNumber;
        total += retstat.val_bins_i[i];
    }
    // std::cout << "sumcheck " << get<parameters::num_ptcls_fluid>() << " " << total << std::endl;

    return retstat;
}
void calculateAndWriteStatistics(){
    if (get<parameters::internal::num_ptcls_fluid>() == 0)
        return;
    //velocity stats
    float sum = 0.f, sump = 0.f, sumd = 0.f;
    std::vector<float> vels, pres, dens;

    float maxvel = 0;
    for (int i = 0; i < get<parameters::internal::num_ptcls>(); i++)
    {
        if (arrays::particle_type::ptr[i] != 0) continue;
        auto vel = math::length3(arrays::velocity::ptr[i]);
        if (vel < 0) vel = 0;
        if (maxvel < vel) maxvel = vel;
        sum += vel;
        vels.push_back(vel);

        sump += arrays::pressure::ptr[i];
        pres.push_back(arrays::pressure::ptr[i]);

        sumd += arrays::density::ptr[i];
        dens.push_back(arrays::density::ptr[i]);
    }
    vels.push_back(sum);
    pres.push_back(sump);
    dens.push_back(sumd);

    auto velstat = calculateStats(vels, 40.f);
    auto presstat = calculateStats(pres, 1000000.f);
    auto densstat = calculateStats(dens, 1.f);
    
    auto fl = std::fstream(get<parameters::internal::folderName>() + "/" + fulldump, std::ios::app);
    
    fl << (velstat.to_string() + "\t\t" + presstat.to_string() + "\t\t" + densstat.to_string() + "\n");

    fl.close();
    
    // std::cout << "MAXVEL " << maxvel << "maxvelstat " << velstat.max << std::endl;
}

void SPH::logDump::create_log_folder(Memory mem){
    auto folderName = get<parameters::internal::folderName>() + "_" + currentDateTime();
    get<parameters::internal::folderName>() = folderName;

    boost::filesystem::path dir(folderName.c_str());
    if(boost::filesystem::create_directory(dir)) {
		std::cout << "Success" << "\n";
	}

    if(boost::filesystem::create_directory(get<parameters::internal::folderName>() + "/export")) {
		std::cout << "Success2" << "\n";
    }
    std::cout << "current " << boost::filesystem::current_path() << std::endl;

    get<parameters::render_settings::gl_file>() = folderName + "/" + get<parameters::render_settings::gl_file>();
    std::cout << "vide file " << get<parameters::render_settings::gl_file>() << std::endl;
    auto fl = std::fstream(get<parameters::internal::folderName>() + "/" + fulldump, std::ios::app);
    
    fl << "vel_max vel_min vel_avg vel_short_avg vel_xt vel_deviation median\t";
    for (int i = 0; i < range*2+1; i++)
        fl << "vel" << i << (i == range*2 ? "" :" ");
    fl << "\t\t";
    fl << "pres_max pres_min pres_avg pres_short_avg pres_xt pres_deviation median\t";
    for (int i = 0; i < range*2+1; i++)
        fl << "pres" << i << (i == range*2 ? "" :" ");
    fl << "\t\t";
    fl << "dens_max dens_min dens_avg dens_short_avg dens_xt dens_deviation median\t";
    for (int i = 0; i < range*2+1; i++)
        fl << "dens" << i << (i == range*2 ? "" :" ");
    fl << "\n";
        
    fl.close();

    auto fl1 = std::fstream(get<parameters::internal::folderName>() + "/" + rigiddump + ".txt", std::ios::app);
    fl1 << "cnt";
    fl1 << " time";
    fl1 << " numiter";
    fl1 << " fluiderr";
    fl1 << " rigiderr";
    fl1 << " maxdens";
    fl1 << " predicdens";
    //for (int i = 0; i < get<parameters::rigidVolumes>().size(); i++)
    //{
    //    fl1 << " velx" << i << " vely" << i << " velz" << i << " avelx" << i << " avely" << i 
    //        << " avelz" << i << " mcenterx" << i << " mcentery" << i << " mcenterz" << i << " mass" << i;   
    //}
    fl1 << std::endl;
    fl1.close();

    fl1 = std::fstream(get<parameters::internal::folderName>() + "/energy.txt", std::ios::app);
    fl1 << "time";
    fl1 << " fluidkin";
    fl1 << " fluidpot";
    fl1 << " fluidtot";
    //for (int i = 0; i < get<parameters::rigidVolumes>().size(); i++)
    //{
    //    fl1 << " kin" << i << " pot" << i << " tot" << i;   
    //}
    fl1 << std::endl;
    fl1.close();


    // auto fl1 = std::fstream(get<parameters::folderName>() + "/particledump.txt", std::ios::app);
    // fl1 << "num";
    // fl1 << " density";
    // fl1 << " predicdens";
    // fl1 << " diffdens";
    // fl1 << " volume";
    // fl1 << " aparvol";
    // fl1 << " velx";
    // fl1 << " vely";
    // fl1 << " velz";
    // fl1 << " sr1";
    // fl1 << " sr2";
    // fl1 << " svel1x";
    // fl1 << " svel1y";
    // fl1 << " svel1z";
    // fl1 << " svel2x";
    // fl1 << " svel2y";
    // fl1 << " svel2z";
    // fl1 << " diag";
    // fl1 << std::endl;
    // fl1.close();

}

void writePhysics(){
    std::string fulldump = "full_dump";
    auto myfile = std::fstream(get<parameters::internal::folderName>() + "/" + fulldump + "_" + std::to_string(get<parameters::internal::frame>()) + ".bin", std::ios::out | std::ios::binary);
    
    int fuid = 0;
    int nums = 0;
    for (int i = 0; i < get<parameters::internal::num_ptcls>(); i++)
    {   
        if (arrays::particle_type::ptr[i] == 0) nums++;
    }
    std::vector<int> ids;
    myfile.write((char*)&nums, sizeof(int));
    myfile.write((char*)&get<parameters::internal::simulationTime>(), sizeof(float));
    
    // myfile.write((char*)&get<parameters::num_ptcls>(), sizeof(int));
    // std::cout << "nummmm " << nums << std::endl;
    //pressure
    for (int i = 0; i < get<parameters::internal::num_ptcls>(); i++)
    {   
        if (arrays::pressure::ptr[i] != 0 && arrays::particle_type::ptr[i] != 0) ids.push_back(i);
        if (arrays::particle_type::ptr[i] != 0) continue;
        /*if (arrays::uid::ptr[i] == 0)
            std::cout << "pressure " << arrays::pressure::ptr[i] << std::endl;
        */// if (arrays::particle_type::ptr[i] == 0) fuid = i;
        auto len = math::length(arrays::velocity::ptr[i]);
        myfile.write((char*)&arrays::particle_type::ptr[i], sizeof(int));
        myfile.write((char*)&arrays::pressure::ptr[i], sizeof(float));
        myfile.write((char*)&len, sizeof(float));
        myfile.write((char*)&arrays::velocity::ptr[i].x, sizeof(float));
        myfile.write((char*)&arrays::velocity::ptr[i].y, sizeof(float));
        myfile.write((char*)&arrays::velocity::ptr[i].z, sizeof(float));
        myfile.write((char*)&arrays::density::ptr[i], sizeof(float));
        myfile.write((char*)&arrays::position::ptr[i].x, sizeof(float));
        myfile.write((char*)&arrays::position::ptr[i].y, sizeof(float));
        myfile.write((char*)&arrays::position::ptr[i].z, sizeof(float));
        myfile.write((char*)&arrays::uid::ptr[i], sizeof(int));
        myfile.write((char*)&arrays::acceleration::ptr[i].x, sizeof(float));
        myfile.write((char*)&arrays::acceleration::ptr[i].y, sizeof(float));
        myfile.write((char*)&arrays::acceleration::ptr[i].z, sizeof(float));
        //myfile.write((char*)&arrays::kerkersum::ptr[i].x, sizeof(float));
        //myfile.write((char*)&arrays::kerkersum::ptr[i].y, sizeof(float));
        //myfile.write((char*)&arrays::kerkersum::ptr[i].z, sizeof(float));
        // myfile.write((char*)&arrays::uid::ptr[i], sizeof(int));
        // printf("source %f\n", arrays::_sourceTerm::ptr[i]);
        // myfile.write((char*)&arrays::_sourceTerm::ptr[i], sizeof(float));
        // myfile.write((char*)&arrays::ker::ptr[i], sizeof(float));
        // myfile.write((char*)&arrays::dker::ptr[i].x, sizeof(float));
        // myfile.write((char*)&arrays::dker::ptr[i].y, sizeof(float));
        // myfile.write((char*)&arrays::dker::ptr[i].z, sizeof(float));
        // myfile.write((char*)&arrays::acc::ptr[i].x, sizeof(float));
        // myfile.write((char*)&arrays::acc::ptr[i].y, sizeof(float));
        // myfile.write((char*)&arrays::acc::ptr[i].z, sizeof(float));
        // myfile.write((char*)&arrays::kersum::ptr[i], sizeof(float));
        // myfile.write((char*)&arrays::diag::ptr[i], sizeof(float));
        // myfile.write((char*)&arrays::gwob_predictedAcceleration::ptr[i].x, sizeof(float));
        // myfile.write((char*)&arrays::gwob_predictedAcceleration::ptr[i].y, sizeof(float));
        // myfile.write((char*)&arrays::gwob_predictedAcceleration::ptr[i].z, sizeof(float));
        // myfile.write((char*)&arrays::a_pressure::ptr[i], sizeof(float));
        // myfile.write((char*)&arrays::_predictedAcceleration::ptr[i].x, sizeof(float));
        // myfile.write((char*)&arrays::_predictedAcceleration::ptr[i].y, sizeof(float));
        // myfile.write((char*)&arrays::_predictedAcceleration::ptr[i].z, sizeof(float));

    }
    // if (get<parameters::frame>() >= 6982)
    // if (ids.size() > 0)
    // {
    //     std::cout << "vel " << math::length(arrays::velocity::ptr[fuid]) << " pres " << arrays::pressure::ptr[fuid] 
    //     << " dens " << arrays::density::ptr[fuid] << " src " << arrays::_sourceTerm::ptr[fuid] << " kersum " << arrays::kersum::ptr[fuid]
    //     << " diag " << arrays::diag::ptr[fuid] 
    //     << "\n";
    //     for (auto i:ids)
    //     {
    //         std::cout << "vel" << i << " " << math::length(arrays::velocity::ptr[i]) << " pres " << arrays::pressure::ptr[i] 
    //         << " dens " << arrays::density::ptr[i] << " src " << arrays::_sourceTerm::ptr[i] << " kersum " << arrays::kersum::ptr[i]
    //         << " diag " << arrays::diag::ptr[i] 
    //         << "\n";
        
    //     }
    // }
    myfile.close();
}


void writePhysics2(){
    std::string fulldump = "full_dump";
    //auto myfile = std::fstream(get<parameters::internal::folderName>() + "/" + fulldump + "_" + std::to_string(get<parameters::internal::frame>()) + ".csv", std::ios::out | std::ios::binary);
    
    std::ofstream myfile;
    myfile.open (get<parameters::internal::folderName>() + "/" + fulldump + "_" + std::to_string(get<parameters::internal::frame>()) + ".csv");
    myfile << "x,y,z\n";
    
    
    for (int i = 0; i < get<parameters::internal::num_ptcls>(); i++)
    {   
        if (arrays::particle_type::ptr[i] != 0) continue;
        
        myfile << arrays::position::ptr[i].x << "," << arrays::position::ptr[i].y << "," << arrays::position::ptr[i].z << std::endl;
       

    }
    myfile.close();
}
#include <iomanip>

void writeEnergy(){
    auto myfile = std::fstream(get<parameters::internal::folderName>() + "/energy.txt", std::ios::app);
    myfile.precision(10);
    myfile << get<parameters::internal::simulationTime>();
    
    std::vector<float> kinen;
    std::vector<float> poten;

    //for (int i = 0; i < get<parameters::rigidVolumes>().size(); i++)
    //{
    //    kinen.push_back(0.f);
    //    poten.push_back(0.f);
    //}

    for (int i = 0; i < get<parameters::internal::num_ptcls>(); i++)
    {
        if (arrays::particle_type::ptr[i] != 0) continue;
        auto vel = math::length3(arrays::velocity::ptr[i]);
        if (vel < 0) vel = 0;
        kinen[arrays::particle_type::ptr[i]] += vel * vel / 2;
        poten[arrays::particle_type::ptr[i]] += get<parameters::simulation_settings::external_force>().z * arrays::position::ptr[i].z;
        
    }
    // std::cout << "asdfsadff " << get<parameters::external_force>().z * arrays::position::ptr[0].z;
    //for (int i = 0; i < get<parameters::rigidVolumes>().size(); i++)
    //    myfile << " " << kinen[i] << " " << poten[i] << " " << (kinen[i] - poten[i]);
    
    myfile << "\n";
    myfile.close();
}

void writeRigids(){
    //auto myfile = std::fstream(get<parameters::internal::folderName>() + "/" + rigiddump + ".txt", std::ios::app);
    //myfile.precision(10);
    //myfile << get<parameters::rigidVolumes>().size();
    //myfile << " " << get<parameters::internal::simulationTime>();
    ////myfile << " " << get<parameters::internal::iterations>();
    ////myfile << " " << get<parameters::internal::density_error>();
    //

    ////find max rigid density
    //float maxdens = 0.f;
    //int ind = 0;
    //for (int i = 0; i < get<parameters::num_ptcls>(); i++)
    //    if (arrays::particle_type::ptr[i] != 0)
    //        if (arrays::densityRigid::ptr[i] > maxdens) 
    //        {
    //            maxdens = arrays::densityRigid::ptr[i];
    //            ind = i;
    //        }

    //myfile << " " << maxdens;
    //myfile << " " << arrays::predictedDensity::ptr[ind];
    //
    //// std::cout << std::setprecision( 10 );
    ////pressure
    //for (int i = 0; i < get<parameters::rigidVolumes>().size(); i++)
    //{   
    //    // myfile << std::hexfloat;
    //    // myfile << " ";
    //    // myfile << arrays::gwob_rrBodyvelocity::ptr[i].z;
    //    myfile << " ";
    //    myfile << arrays::velocityR::ptr[i].x;
    //    myfile << " ";
    //    myfile << arrays::velocityR::ptr[i].y;
    //    myfile << " ";
    //    myfile << arrays::velocityR::ptr[i].z;

    //    myfile << " ";
    //    myfile << arrays::angvelR::ptr[i].x;
    //    myfile << " ";
    //    myfile << arrays::angvelR::ptr[i].y;
    //    myfile << " ";
    //    myfile << arrays::angvelR::ptr[i].z;
    //    myfile << " ";

    //    myfile << arrays::massCenterR::ptr[i].x;
    //    myfile << " ";
    //    myfile << arrays::massCenterR::ptr[i].y;
    //    myfile << " ";
    //    myfile << arrays::massCenterR::ptr[i].z;
    //    myfile << " ";
    //    myfile << (1.f/arrays::invmassR::ptr[i]);
    //}
    //myfile << std::endl;
    //myfile.close();
}

void SPH::logDump::log_dump(Memory mem){
    // std::cout << "bla1\n";
    writePhysics();
    //writePhysics2();
    
    //writeEnergy();
    // std::cout << "bla2\n";
    //calculateAndWriteStatistics();
    // std::cout << "bla3\n";
    //writeRigids();
    // std::cout << "bla4\n";
}