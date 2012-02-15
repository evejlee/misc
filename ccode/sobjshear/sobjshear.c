#include <stdlib.h>
#include <stdio.h>
#include "config.h"
#include "source.h"
#include "shear.h"
#include "log.h"
#include "defs.h"

void usage_and_exit(void) {
    wlog("usage: sobjshear config_url lens_cat\n");
    exit(EXIT_FAILURE);
}


int main(int argc, char** argv) {
    int64 counter=0;

    if (argc < 3) {
        usage_and_exit();
    }

    const char* config_url=argv[1];
    const char* lens_url=argv[2];
    struct shear* shear=shear_init(config_url, lens_url);

#ifndef WITH_TRUEZ
    struct source* src=source_new(shear->config->nzl);
    src->zlens = shear->config->zl;
#else
    struct source* src=source_new();
#endif

    while (source_read(stdin, src)) {
        counter++;
        if (counter == 1) {
            wlog("first source:\n");
            source_print(src);
        }
        if ((counter % 10000) == 0) {
            wlog(".");
        }

#ifdef WITH_TRUEZ
        src->dc = Dc(shear->cosmo, 0.0, src->z);
#endif

        shear_process_source(shear, src);
    }
    wlog("\nlast source:\n");
    source_print(src);

    wlog("Read a total of %lu sources\n", counter);

    // print some summary info
    shear_print_sum(shear);

    wlog("Writing results to stdout\n");
    shear_write(shear, stdout);

    src=source_delete(src);
    shear=shear_delete(shear);
    wlog("Done\n");

    return 0;
}