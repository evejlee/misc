module cat;
import std.stdio;
static import std.c.stdio;
import std.math;
import point;
import stack;
import healpix;
import hpoint;
import match;

class CatPoint : Point {
    // we inherit x,y,z from Point
    long index;
    double cos_radius;

    this(double ra, double dec, long index, double cos_radius) {
        this.index=index;
        this.cos_radius = cos_radius;
        super(ra,dec);
    }
}

class Cat {

    // a dictionary keyed by longs (htmid) and with values a stack of Points
    Stack!(CatPoint)[long] pdict;
    Healpix hpix;

    this(string filename, long nside, double rad_arcsec) {
        File* file;
        double ra,dec,rad_radians,cos_radius;
        file = new File(filename,"r");

        int rad_in_file=_radius_check(rad_arcsec,&rad_radians,&cos_radius);

        Stack!(long) pixlist;
        hpix = new Healpix(nside);
        long index=0;

        while (2 == std.c.stdio.fscanf(file.getFP(),"%lf %lf\n", &ra, &dec)) {

            if (rad_in_file) {
                fscanf(file.getFP(),"%lf", &rad_arcsec);
                rad_radians = rad_arcsec/3600.*D2R;
                cos_radius = cos(rad_radians);
            }

            auto p = new CatPoint(ra,dec,index,cos_radius);

            hpix.disc_intersect(p, rad_radians, &pixlist);

            for (size_t i=0; i<pixlist.length; i++) {
                long pix=pixlist[i];
                auto cps = (pix in pdict);
                if (cps) {
                    cps.push(p);
                } else {
                    // all the time is spent here
                    // creation of the stack is about 400ms out of 4s, so
                    // factor of 10 quicker.  Insertion is the slow part
                    // docs say insertion may invoke the garbage collector
                    // I tried implementing a tree but it was even slower
                    // also because of GC
                    // maybe we could use a hash from C, but then what's
                    // the point?
                    pdict[pix] = Stack!(CatPoint)(p);
                }
            }
            index++;
        }
    }

    void match(HPoint pt, Stack!(Match)* matches) {
        matches.resize(0);
        auto hpixid = hpix.pixelof(pt);
        writeln(hpixid);
        auto idstack = (hpixid in pdict);
        if (idstack) {
            foreach (cat_point; *idstack) {
                double cos_angle = cat_point.dot(pt);
                if (cos_angle > cat_point.cos_radius) {
                    matches.push(Match(cat_point.index,cos_angle));
                }
            }
        }
    }

    void print_counts() {
        foreach (pix, cps; pdict) {
            writefln("%s %s", pix, cps.length);
        }
    }
    int _radius_check(double radius_arcsec, 
                      double* rad_radians,
                      double* cos_radius)
    {
        int radius_in_file=0;
        if (radius_arcsec <= 0) {
            radius_in_file=1;
        } else {
            radius_in_file=0;
            *rad_radians = radius_arcsec/3600.*D2R;
            *cos_radius = cos(*rad_radians);
        }
        return radius_in_file;
    }

}

unittest
{
    long nside=4096;
    double rad_arcsec=2; 
    string f="/home/esheldon/tmp/rand-radec-10000.dat";
    auto c = new Cat(f,nside,rad_arcsec);

    //c.print_counts();
}
