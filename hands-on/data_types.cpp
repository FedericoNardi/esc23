# include <iostream>
# include <vector>
# include <cstdlib>
# include <chrono>

static double x_max = 1.;
static int _NOBJ;

class ParticleAoS{
        std::string _name;
        double      _x, _y, _z;
        double      _px, _py, _pz;
        float       _mass;
        float       _energy;
        const int   _PID = 0;
        static int  _NOBJ;
        public:
            double  GetX(){return _x;};
            double  GetY(){return _y;};
            double  GetZ(){return _z;};
            double  GetPx(){return _px;};
            double  GetPy(){return _py;};
            double  GetPz(){return _pz;};
            float   GetMass(){return _mass;};
            float   GetEnergy(){return _energy;};
            void SetX(double x){_x=x;};
            void SetY(double y){_y=y;};
            void SetZ(double z){_z=z;};
            void SetPx(double px){_px=px;};
            void SetPy(double py){_py=py;};
            void SetPz(double pz){_pz=pz;};
            void SetMass(float mass){_mass=mass;};
            void SetEnergy(float energy){_energy=energy;};
};

std::vector<ParticleAoS> MakeParticlesAoS(int N)
{
    std::vector<ParticleAoS> GoodParticles(N);
    for(int ii=0; ii<N; ii++)
    {
        GoodParticles[ii].SetX(std::rand()/RAND_MAX);
        GoodParticles[ii].SetY(std::rand()/RAND_MAX);
        GoodParticles[ii].SetZ(std::rand()/RAND_MAX);
        GoodParticles[ii].SetPx(std::rand()/RAND_MAX);
        GoodParticles[ii].SetPy(std::rand()/RAND_MAX);
        GoodParticles[ii].SetPz(std::rand()/RAND_MAX);
        GoodParticles[ii].SetMass(std::rand()/RAND_MAX);
        GoodParticles[ii].SetEnergy(std::rand()/RAND_MAX);
    }
    return GoodParticles;
}

class ParticleSoA{
        std::vector<std::string> _names;
        std::vector<double> _x;
        std::vector<double> _y;
        std::vector<double> _z;
        std::vector<double> _px;
        std::vector<double> _py;
        std::vector<double> _pz;
        std::vector<float>  _mass;
        std::vector<float>  _energy; 
        const std::vector<int> PID;
        public:
            ParticleSoA(int N) 
            {
                _NOBJ=N;
                _x.push_back(std::rand()/RAND_MAX);
                _y.push_back(std::rand()/RAND_MAX);
                _z.push_back(std::rand()/RAND_MAX);
                _px.push_back(std::rand()/RAND_MAX);
                _py.push_back(std::rand()/RAND_MAX);
                _pz.push_back(std::rand()/RAND_MAX);
                _mass.push_back(std::rand()/RAND_MAX);
                _energy.push_back(std::rand()/RAND_MAX);
            }
            // int GetNObjects(){return _NOBJ;};
            std::vector<double> GetX(){return _x;};
            std::vector<double> GetPx(){return _px;};
            std::vector<float>  GetMass(){return _mass;}
            void                SetPx(int idx, double val){_px[idx]=val;}

};

// std::malloc(std::size_t size);

void EvolveSoA(ParticleSoA, double);
void EvolveAoS(std::vector<ParticleAoS>, double);
void FillSoA(void*, int);

int main(){
    /*
    {
        ParticleSoA myParticleSoA{1000};
        auto start = std::chrono::high_resolution_clock::now();
        for(int _=0; _<1000; _++)
        {
            EvolveSoA(myParticleSoA, 0.01);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto const time = end-start;
        auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(time).count();
        std::cout <<"Time elapsed generating SoA: "<<duration<<"s\n";
    }
    {
        auto myParticleAoS = MakeParticlesAoS(1000);
        auto start = std::chrono::high_resolution_clock::now();
        for(int _ = 0; _<1000; _++)
        {
            EvolveAoS(myParticleAoS, 0.01);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto const time = end-start;
        auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(time).count();
        std::cout <<"Time elapsed generating AoS: "<<duration<<"s\n";
    }
    */
    {
        _NOBJ = 10;
        auto SoAptr = std::malloc((6*8+2*4)*_NOBJ); // anche reinterpret_cast
        FillSoA(SoAptr, _NOBJ);
        std::free(SoAptr);
    }
}

void FillSoA(void* ptr, int nobj){
    for(int ii=0; ii<nobj; ii++){
        /*
        auto d_ptr = static_cast<double*>(ptr);
        for(int i_double=0; i_double<6; i_double++){ // NOTA: double*+ sposta di 8; char*+ sposta di 1
            *(d_ptr+i_double*sizeof(char)+ii) = static_cast<double>(std::rand())/static_cast<double>(RAND_MAX);
            std::cout<<*(d_ptr+i_double+ii)<<"\n";
        }
        */
       
        auto d_ptr = static_cast<char*>(ptr);
        for(int i_double=0; i_double<6; i_double++){ // NOTA: double*+ sposta di 8; char*+ sposta di 1
            auto myptr = reinterpret_cast<double*>(d_ptr+i_double*sizeof(double)+ii);
            *myptr = static_cast<double>(std::rand())/static_cast<double>(RAND_MAX);
            std::cout<<*(reinterpret_cast<double*>(d_ptr+i_double*sizeof(double)+ii))<<"\n";
        }

    }
}

void EvolveAoS(std::vector<ParticleAoS> myAoS, double t_step)
{
    for(auto particle : myAoS){
        bool hit_x = false;
        auto new_x = particle.GetX() + particle.GetPx()/(particle.GetMass()+t_step);
        if( new_x<0||new_x>x_max ){
            hit_x = true;
            particle.SetPx(-particle.GetPx());
        }
    }
}

void EvolveSoA(ParticleSoA mySoA, double t_step){
    for(int ii=0; ii<_NOBJ; ii++){
        bool hit_x = false;
        auto new_x =mySoA.GetX()[ii] + mySoA.GetPx()[ii]/(mySoA.GetMass()[ii]+t_step);
        if( new_x<0||new_x>x_max ){
            hit_x = true;
            mySoA.SetPx(ii,-mySoA.GetPx()[ii]);
        }
    }
}