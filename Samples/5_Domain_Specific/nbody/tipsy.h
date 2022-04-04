#ifndef __TIPSY_H__
#define __TIPSY_H__

#include <string>

using namespace std;

#define MAXDIM 3

typedef float Real;

struct gas_particle
{
    Real mass;
    Real pos[MAXDIM];
    Real vel[MAXDIM];
    Real rho;
    Real temp;
    Real hsmooth;
    Real metals ;
    Real phi ;
} ;

//struct gas_particle *gas_particles;

struct dark_particle
{
    Real mass;
    Real pos[MAXDIM];
    Real vel[MAXDIM];
    Real eps;
    int phi ;
} ;

//struct dark_particle *dark_particles;

struct star_particle
{
    Real mass;
    Real pos[MAXDIM];
    Real vel[MAXDIM];
    Real metals ;
    Real tform ;
    Real eps;
    int phi ;
} ;

//struct star_particle *star_particles;

struct dump
{
    double time ;
    int nbodies ;
    int ndim ;
    int nsph ;
    int ndark ;
    int nstar ;
} ;

typedef struct dump header ;

template <typename real4>
void read_tipsy_file(vector<real4> &bodyPositions,
                     vector<real4> &bodyVelocities,
                     vector<int> &bodiesIDs,
                     const std::string &fileName,
                     int &NTotal,
                     int &NFirst,
                     int &NSecond,
                     int &NThird)
{
    /*
       Read in our custom version of the tipsy file format written by
       Jeroen Bedorf.  Most important change is that we store particle id on the
       location where previously the potential was stored.
    */

    char fullFileName[256];
    sprintf(fullFileName, "%s", fileName.c_str());

    cout << "Trying to read file: " << fullFileName << endl;

    ifstream inputFile(fullFileName, ios::in | ios::binary);

    if (!inputFile.is_open())
    {
        cout << "Can't open input file \n";
        exit(EXIT_SUCCESS);
    }

    dump  h;
    inputFile.read((char *)&h, sizeof(h));

    int idummy;
    real4 positions;
    real4 velocity;


    //Read tipsy header
    NTotal        = h.nbodies;
    NFirst        = h.ndark;
    NSecond       = h.nstar;
    NThird        = h.nsph;

    //Start reading
    int particleCount = 0;

    dark_particle d;
    star_particle s;

    for (int i=0; i < NTotal; i++)
    {
        if (i < NFirst)
        {
            inputFile.read((char *)&d, sizeof(d));
            velocity.w        = d.eps;
            positions.w       = d.mass;
            positions.x       = d.pos[0];
            positions.y       = d.pos[1];
            positions.z       = d.pos[2];
            velocity.x        = d.vel[0];
            velocity.y        = d.vel[1];
            velocity.z        = d.vel[2];
            idummy            = d.phi;
        }
        else
        {
            inputFile.read((char *)&s, sizeof(s));
            velocity.w        = s.eps;
            positions.w       = s.mass;
            positions.x       = s.pos[0];
            positions.y       = s.pos[1];
            positions.z       = s.pos[2];
            velocity.x        = s.vel[0];
            velocity.y        = s.vel[1];
            velocity.z        = s.vel[2];
            idummy            = s.phi;
        }

        bodyPositions.push_back(positions);
        bodyVelocities.push_back(velocity);
        bodiesIDs.push_back(idummy);

        particleCount++;
    }//end for

    // round up to a multiple of 256 bodies since our kernel only supports that...
    int newTotal = NTotal;

    if (NTotal % 256)
    {
        newTotal = ((NTotal / 256) + 1) * 256;
    }

    for (int i = NTotal; i < newTotal; i++)
    {
        positions.w = positions.x = positions.y = positions.z = 0;
        velocity.x = velocity.y = velocity.z = 0;
        bodyPositions.push_back(positions);
        bodyVelocities.push_back(velocity);
        bodiesIDs.push_back(i);
        NFirst++;
    }

    NTotal = newTotal;

    inputFile.close();

    cerr << "Read " << NTotal << " bodies" << endl;
}

#endif //__TIPSY_H__
