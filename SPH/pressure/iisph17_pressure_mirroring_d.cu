#include <SPH/pressure/iisph17_pressure_mirroring_d.cuh>
#include <utility/include_all.h>

bool isFluid(int type) {
    return type == 0 ? true : false;
}
bool isRigid(int type) {
    return type != 0 ? true : false;
}

neighFunctionType IISPH_predictAdvection2(SPH::IISPH17_PRESSURE_MIRRORING_D::Memory arrays) {
    checkedParticleIdx(i);
    cache_arrays((pos, position));

    //if (arrays.particle_type[i] == 0) 
        //printf("init vel  %f %f %f\n", arrays.velocity[i].val.x, arrays.velocity[i].val.y, arrays.velocity[i].val.z);
    arrays.velocity[i] = arrays.velocity[i] + arrays.timestep * arrays.acceleration[i];
    arrays._apparentVolume[i] = arrays.volume[i] / arrays.density[i];
    return;
}



neighFunctionType IISPH_prepareSolver2(SPH::IISPH17_PRESSURE_MIRRORING_D::Memory arrays) {
    checkedParticleIdx(i);
    cache_arrays((pos, position), (v_adv, velocity), (i_vol, _apparentVolume), (vol, volume));
    // auto s_i = 1.f / arrays.density[i] - 1.f;
    auto s_i = 1.f - vol[i] / i_vol[i];
    /*if (arrays.particle_type[i] == 0)
        printf("source1 %f\n", s_i);*/
     //s_i = 0.f;
    ////auto oldsi = s_i;
    //if (arrays.particle_type[i] == 0)
    //    printf("s_1 %f\n", s_i);
    float a_ii = 0.f;
    float4 kernelSum = { 0.f, 0.f, 0.f, 0.f };
    float gradtmp = 0.f;

    iterateNeighbors(j) {
        auto spikyGradient = GPW_ij;

        kernelSum += (i_vol[j] * spikyGradient).val;
        if (arrays.particle_type[j] == 0)
            a_ii += (i_vol[j] * i_vol[j] / (arrays.rest_density * vol[j]) * math::dot3(spikyGradient, spikyGradient)).val;

        if (arrays.particle_type[i] == 0 || (arrays.particle_type[i] != 0 && arrays.particle_type[j] == 0)) {
            auto srctmp = arrays.timestep * i_vol[j] * math::dot3(v_adv[i] - (arrays.particle_type[j] == 0 ? v_adv[j] : 0), spikyGradient);
            s_i = s_i - srctmp;
            //if (arrays.particle_type[i] == 0) printf("srctmp %.30f vol %.30f\n", srctmp.val, i_vol[j].val);
            
        }
    }
   /* if (arrays.particle_type[i] == 0)
        printf("s_2 %f posz %f\n", s_i, pos[i].val.z);*/

    if (arrays.particle_type[i] == 0)
        a_ii += math::dot3(kernelSum, kernelSum) / (vol[i] * arrays.rest_density);

    arrays._sourceTerm[i] = s_i;
 /*   if (arrays.particle_type[i] == 0)
        printf("source2 %f %.30f\n", s_i, gradtmp);*/
    arrays._Aii[i].val = (-math::square(arrays.timestep) * i_vol[i] * a_ii).val;
    arrays.pressure.second[i] = 0.f;
}


neighFunctionType IISPH_jacobiFirst2(const SPH::IISPH17_PRESSURE_MIRRORING_D::Memory arrays) {
    checkedParticleIdx(i);
    cache_arrays((pos, position), (i_vol, _apparentVolume), (pressure, pressure.second));

    const auto factor = -1.f * i_vol[i] / (arrays.volume[i] * arrays.rest_density);

    float4 kernelSum = { 0.f, 0.f, 0.f, 0.f };
    float4 kerker = { 0.f, 0.f, 0.f, 0.f };
    float poses =  0.f;
    if (arrays.particle_type[i] == 0)
        iterateNeighbors(j)
    {
        if (arrays.particle_type[j] == 0)
            kernelSum += (factor * i_vol[j] * (pressure[i] + pressure[j]) * GPW_ij).val;
        if (arrays.particle_type[j] != 0)
        {
            kernelSum += (factor * i_vol[j] * (pressure[i] + pressure[i]) * GPW_ij).val;
            kerker += GPW_ij.val;
            auto df = pos[i] - pos[j];
            float partxy = df.val.x * df.val.x + df.val.y * df.val.y;
            auto d = math::length3(df).val;
            poses += (pos[i].val.w - d)*(pos[i].val.w - d)/d * df.val.z;
            //poses += partxy;
        }
    }

    //arrays.kerkersum[i] = kerker;
    arrays._predictedAcceleration[i].val = kernelSum;

    //if (arrays.particle_type[i] == 0) {
    //    auto acctmp = arrays.acceleration[i].val.z + arrays._predictedAcceleration[i].val.z;
    //    printf("kerz %f posz %f pres %f vel %f acc %c%f dns %f src %.12f\n", kerker.z, pos[i].val.z, arrays.pressure.second[i].val, (arrays.velocity[i] - arrays.timestep * arrays.acceleration[i]).val.z, 
    //        acctmp > 0 ? '+' : '-', abs(acctmp), arrays.density[i].val, arrays._sourceTerm[i].val);
    //    //arrays.position[i].val.z -= 0.01;
    //}
    //if (arrays.particle_type[i] == 0 && kernelSum.x != 0.f) printf("kernel %f %f %f factor %.30f bool %d\n", kernelSum.x, kernelSum.y, kernelSum.z, factor.val, factor.val == 0.0 ? 1 : 0);

    //========================== PRESSURE SWAP ========================================
    arrays.pressure.first[i] = arrays.pressure.second[i];
}

neighFunctionType IISPH_jacobiSecond2(const SPH::IISPH17_PRESSURE_MIRRORING_D::Memory arrays) {
    checkedParticleIdx(i);
    cache_arrays((pos, position), (acc, _predictedAcceleration));
    alias_arrays((vol, volume), (i_vol, _apparentVolume));

    if (math::abs(arrays._Aii[i]) < 1e-20f) {
        arrays.pressure.second[i] = 0._Pa;
        return;
    }

    float kernelSum = 0.f;

    //printf("000");
    if (arrays.particle_type[i] == 0)
    {
        //printf("111");
        iterateNeighbors(j)
            // kernelSum += (math::square(arrays.timestep) * i_vol[j] * math::dot3(acc[i] - acc[j], GPW_ij)).val;
            kernelSum += (math::square(arrays.timestep) * i_vol[j] * math::dot3(acc[i] - (arrays.particle_type[j] == 0 ? acc[j] : 0), GPW_ij)).val;
    }

    // TODO: set different omega for fluid and rigid particles
    float omega = 0.5f;

    //if (arrays.particle_type[i] != 0) omega = omega * vol[i].val / 0.001;

    auto pressure = (math::max(arrays.pressure.first[i] + omega * (arrays._sourceTerm[i].val - kernelSum) / arrays._Aii[i], 0.f)) + 0._Pa;

    auto residual = (kernelSum - arrays._sourceTerm[i].val) * 1._Pa;

    if (math::abs(arrays._Aii[i]) < 1e-20f || pressure != pressure || pressure > 1e16f || pressure < 0.f) {
        pressure = 0._Pa;
        residual = decltype(residual)(0.f);
    }
    
    arrays.pressure.second[i] = pressure;

    /*if (pressure.val != 0)
    {
        printf("pressure %f nei %d\n", pressure.val, arrays.neighborListLength[i]);
    }*/

    /*if (arrays.particle_type[i] == 0 && pressure.val != 0.f)
       printf("pressure %f - dif %.12f - A %.12f - dens %f - accz %f\n", pressure.val, (arrays._sourceTerm[i].val - kernelSum), arrays._Aii[i].val, 
           arrays.density[i].val, arrays._predictedAcceleration[i].val.z);*/
    arrays._volumeError[i] = math::max(residual, 0.f).val * arrays.volume[i].val;
    if (arrays.particle_type[i] != 0) arrays._volumeError[i] = 0.f;
}


basicFunctionType updateAccelerations2(SPH::IISPH17_PRESSURE_MIRRORING_D::Memory arrays) {
    checkedParticleIdx(i);

    if (arrays.particle_type[i] != 0) {
        //printf("type %d\n", arrays.particle_type[i]);
        return;
    }

    //printf("accccREal %.20f\n", arrays._predictedAcceleration[i].val.z);
    arrays.velocity[i] = arrays.velocity[i] - arrays.timestep * arrays.acceleration[i];
    if (math::length3(arrays.velocity[i].val) > 100.f) arrays.velocity[i].val /= math::length3(arrays.velocity[i].val);
    //printf("return vel  %f %f %f\n", arrays.velocity[i].val.x, arrays.velocity[i].val.y, arrays.velocity[i].val.z);
    arrays.acceleration[i] += arrays._predictedAcceleration[i];
    //if (math::length3(arrays.acceleration[i].val) > 100) arrays.acceleration[i].val /= math::length3(arrays.acceleration[i].val);
    /*printf("pred acc %f %f %f\n", arrays._predictedAcceleration[i].val.x, arrays._predictedAcceleration[i].val.y, arrays._predictedAcceleration[i].val.z);
    printf("final acc %f %f %f\n", arrays.acceleration[i].val.x, arrays.acceleration[i].val.y, arrays.acceleration[i].val.z);*/
}


neighFunction(Predict, IISPH_predictAdvection2, "IISPH17_PRESSURE_MIRRORING_D: predict Advection", caches<float4>{});
neighFunction(Prepare, IISPH_prepareSolver2, "IISPH17_PRESSURE_MIRRORING_D: prepare Solver", caches<float4, float4, float, float, float>{});
neighFunction(Jacobi1, IISPH_jacobiFirst2, "IISPH17_PRESSURE_MIRRORING_D: jacobi First", caches<float4, float, float, float>{});
neighFunction(Jacobi2, IISPH_jacobiSecond2, "IISPH17_PRESSURE_MIRRORING_D: jacobi Second", caches<float4, float4, float>{});
basicFunction(Update, updateAccelerations2, "IISPH17_PRESSURE_MIRRORING_D: updaate Acceleration");

float err(SPH::IISPH17_PRESSURE_MIRRORING_D::Memory mem) {
    float errfluid = 0.f;
    int fnum = 0;
    for (int i = 0; i < mem.num_ptcls; i++)
    {
        if (mem.particle_type[i] == 0)
        {
            fnum++;
            errfluid += mem._volumeError[i].val;
        }
    }
    return errfluid / fnum;
}
void SPH::IISPH17_PRESSURE_MIRRORING_D::pressure_solve(Memory mem) {

    bool test = true;
    size_t iteration = 0;
    float limit = 0.000001;
    std::cout << "============================================================= LIMIT " << limit << "\n";
    std::cout << "gissler mirror " << limit << " \n";
    //mem.velocity[0] = mem.velocity[0] + mem.timestep * mem.acceleration[0];
    launch<Predict>(mem.num_ptcls, mem); //velocity, _apparentVolume
    launch<Prepare>(mem.num_ptcls, mem); //Aii, sourceTerm, swap pressures
    float density_error = 0.f;
    do {

        launch<Jacobi1>(mem.num_ptcls, mem); //predictedAcceleration, swap pressure
        launch<Jacobi2>(mem.num_ptcls, mem); //pressure, volumeError
        //density_error = err(mem);
        density_error = math::getValue(algorithm::reduce_sum(mem._volumeError, mem.num_ptcls) / mem.num_ptcls_fluid);
        test = iteration < 5;
        test = test || density_error > limit;
        test = test && (iteration < 6);
        iteration++;
    } while (test);
     std::cout << "iter " << iteration << " err " << density_error << std::endl;
    //std::cout << iteration << " -> " << errfluid << std::endl;

    launch<Jacobi1>(mem.num_ptcls, mem);
    launch<Update>(mem.num_ptcls, mem);

    //get<parameters::iterations>() = (int32_t)iteration;
    // printf("---------------------------------------------------------- rest: %f\n\n", mem.rest_density.val);
}

void SPH::IISPH17_PRESSURE_MIRRORING_D::print_velocity(Memory mem) {
    for (int i = 0; i < mem.num_ptcls; i++)
    {
        if (mem.particle_type[i] != 0)
        {
            auto vel = mem.velocity[i].val;
            std::cout << "vel " << vel.x << " " << vel.y << " " << vel.z << std::endl;
        }
    }
            std::cout << "vel -------------------" << std::endl;
}

void SPH::IISPH17_PRESSURE_MIRRORING_D::print_begin(Memory mem) {
    
    std::cout << "begin :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::" << std::endl;
}