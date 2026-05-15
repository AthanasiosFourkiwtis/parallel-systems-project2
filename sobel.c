/*
 * MYE023 - Ergasia 2 - Askisi 2
 * Sobel filter sti GPU me OpenMP target offloading
 * Fourkiotis Athanasios, 4940
 *
 * compile:
 *   clang -O2 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
 *     -DNUM_TEAMS=30 -DTHREAD_LIMIT=128 sobel.c -lm -o sobel
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

/* defaults, sto script ta vazw apo to compile gia kathe peirama */
#ifndef NUM_TEAMS
#define NUM_TEAMS 30
#endif

#ifndef THREAD_LIMIT
#define THREAD_LIMIT 128
#endif

#pragma pack(push, 2)  /*Αυτό είναι δομή που κρατά πληροφορίες για BMP αρχείο.*/
typedef struct bmpheader_ {
    char sign;
    int size;
    int notused;
    int data;
    int headwidth;
    int width;
    int height;
    short numofplanes;
    short bitpix;
    int method;
    int arraywidth;
    int horizresol;
    int vertresol;
    int colnum;
    int basecolnum;
} bmpheader_t;
#pragma pack(pop)

/* Η δομή img_t κρατάει και τα raw BMP δεδομένα και τα τρία RGB κανάλια ξεχωριστά. Το Sobel εφαρμόζεται ξεχωριστά σε red, green και blue. */
typedef struct img_ {
    bmpheader_t header;
    int rgb_width;
    unsigned char *imgdata;
    unsigned char *red;
    unsigned char *green;
    unsigned char *blue;
} img_t;

void sobel_serial(img_t *, img_t *);
void sobel_omp_device(img_t *, img_t *);
void sobel_omp_device_nested(img_t *, img_t *);

/* ----- BMP utilities (apo ton dosmeno kwdika, prosthesa kapoia error checks) ----- */

static
void bmp_read_img_from_file(char *inputfile, img_t *img)
/*Η bmp_read_img_from_file διαβάζει την εικόνα BMP από αρχείο, ελέγχει ότι είναι 24-bit και φορτώνει τα raw pixel data στη μνήμη.*/
{
    FILE *file;
    bmpheader_t *header = &(img->header);

    file = fopen(inputfile, "rb");
    if (file == NULL)
    {
        fprintf(stderr, "File %s not found; exiting.", inputfile);
        exit(1);
    }

    if (fread(header, sizeof(bmpheader_t)+1, 1, file) != 1)
    {
        fprintf(stderr, "Cannot read BMP header from %s; exiting.", inputfile);
        fclose(file);
        exit(1);
    }
    if (header->bitpix != 24)
    {
        fprintf(stderr, "File %s is not in 24-bit format; exiting.", inputfile);
        fclose(file);
        exit(1);
    }
    if (header->width <= 0 || header->height <= 0 || header->arraywidth <= 0)
    {
        fprintf(stderr, "File %s has invalid BMP dimensions; exiting.", inputfile);
        fclose(file);
        exit(1);
    }

    img->imgdata = (unsigned char*) calloc(header->arraywidth, sizeof(unsigned char));
    if (img->imgdata == NULL)
    {
        fprintf(stderr, "Cannot allocate memory for image data; exiting.");
        fclose(file);
        exit(1);
    }

    if (fseek(file, header->data, SEEK_SET) != 0)
    {
        fprintf(stderr, "Cannot seek to BMP data in %s; exiting.", inputfile);
        free(img->imgdata);
        fclose(file);
        exit(1);
    }
    if (fread(img->imgdata, header->arraywidth, 1, file) != 1)
    {
        fprintf(stderr, "Cannot read BMP pixel data from %s; exiting.", inputfile);
        free(img->imgdata);
        fclose(file);
        exit(1);
    }
    fclose(file);
}

static
void bmp_clone_empty_img(img_t *imgin, img_t *imgout)
{
	/*Η bmp_clone_empty_img δημιουργεί μία κενή εικόνα εξόδου με το ίδιο μέγεθος και header με την είσοδο.*/
    imgout->header = imgin->header;
    imgout->imgdata =
        (unsigned char*) calloc(imgout->header.arraywidth, sizeof(unsigned char));
    if (imgout->imgdata == NULL)
    {
        fprintf(stderr, "Cannot allocate memory for clone image data; exiting.");
        exit(1);
    }
}

static
void bmp_write_data_to_file(char *fname, img_t *img)
{
    FILE *file;
    bmpheader_t *bmph = &(img->header);

    file = fopen(fname, "wb");
    if (file == NULL)
    {
        fprintf(stderr, "Cannot open %s for writing; exiting.", fname);
        exit(1);
    }
    if (fwrite(bmph, sizeof(bmpheader_t)+1, 1, file) != 1)
    {
        fprintf(stderr, "Cannot write BMP header to %s; exiting.", fname);
        fclose(file);
        exit(1);
    }
    if (fseek(file, bmph->data, SEEK_SET) != 0)
    {
        fprintf(stderr, "Cannot seek to BMP data in %s; exiting.", fname);
        fclose(file);
        exit(1);
    }
    if (fwrite(img->imgdata, bmph->arraywidth, 1, file) != 1)
    {
        fprintf(stderr, "Cannot write BMP pixel data to %s; exiting.", fname);
        fclose(file);
        exit(1);
    }
    fclose(file);
}

static
void bmp_rgb_from_data(img_t *img)
{
	/*Η bmp_rgb_from_data μετατρέπει τα raw δεδομένα της εικόνας σε τρεις ξεχωριστούς πίνακες RGB.*/
    bmpheader_t *bmph = &(img->header);

    int i, j, pos = 0;
    int width = bmph->width, height = bmph->height;
    int rgb_width = img->rgb_width;

    for (i = 0; i < height; i++)
        for (j = 0; j < width * 3; j += 3, pos++)
        {
            img->red[pos]   = img->imgdata[i * rgb_width + j];
            img->green[pos] = img->imgdata[i * rgb_width + j + 1];
            img->blue[pos]  = img->imgdata[i * rgb_width + j + 2];
        }
}

static
void bmp_data_from_rgb(img_t *img)
{
	/*Η bmp_data_from_rgb ξαναενώνει τα RGB κανάλια σε raw BMP δεδομένα για να αποθηκευτεί η εικόνα.*/
    bmpheader_t *bmph = &(img->header);
    int i, j, pos = 0;
    int width = bmph->width, height = bmph->height;
    int rgb_width = img->rgb_width;

    for (i = 0; i < height; i++ )
        for (j = 0; j < width* 3 ; j += 3 , pos++)
        {
            img->imgdata[i * rgb_width  + j]     = img->red[pos];
            img->imgdata[i * rgb_width  + j + 1] = img->green[pos];
            img->imgdata[i * rgb_width  + j + 2] = img->blue[pos];
        }
}

static
void bmp_rgb_alloc(img_t *img)
{
	/*Η bmp_rgb_alloc δεσμεύει μνήμη για τα τρία κανάλια RGB, ώστε κάθε κανάλι να είναι ξεχωριστός πίνακας μεγέθους width × height.*/
    int width, height;

    width = img->header.width;
    height = img->header.height;

    img->red = (unsigned char*) calloc(width*height, sizeof(unsigned char));
    if (img->red == NULL)
    {
        fprintf(stderr, "Cannot allocate memory for the red channel; exiting.");
        exit(1);
    }

    img->green = (unsigned char*) calloc(width*height, sizeof(unsigned char));
    if (img->green == NULL)
    {
        fprintf(stderr, "Cannot allocate memory for the green channel; exiting.");
        free(img->red);
        exit(1);
    }

    img->blue = (unsigned char*) calloc(width*height, sizeof(unsigned char));
    if (img->blue == NULL)
    {
        fprintf(stderr, "Cannot allocate memory for the blue channel; exiting.");
        free(img->red);
        free(img->green);
        exit(1);
    }

    img->rgb_width = width * 3;
    if ((width * 3  % 4) != 0) {
       img->rgb_width += (4 - (width * 3 % 4));
    }
}

static
void bmp_img_free(img_t *img)
{
    free(img->red);
    free(img->green);
    free(img->blue);
    free(img->imgdata);
}

/* ----- telos BMP utilities ----- */

/* Η clamp χρειάζεται για τα pixels στα όρια της εικόνας. Αν η γειτονιά 3×3 βγει έξω από την εικόνα, περιορίζω τις συντεταγμένες μέσα στα έγκυρα όρια. */
int clamp(int i , int min , int max)
{
    if (i < min) return min;
    else if (i > max) return max;
    return i;
}

/* seiriako Sobel - opws dothike, den to peiraxa */
void sobel_serial(img_t *imgin, img_t *imgout)
{
    int width = imgin->header.width;
    int height = imgin->header.height;
    int rx = 0, ry = 0, gx = 0, gy = 0, bx = 0, by = 0, idx = 0;

    unsigned char *in_r = imgin->red;
    unsigned char *in_g = imgin->green;
    unsigned char *in_b = imgin->blue;

    unsigned char *out_r = imgout->red;
    unsigned char *out_g = imgout->green;
    unsigned char *out_b = imgout->blue;

    int Gx[9] = {
        -1,  0,  1,
        -2,  0,  2,
        -1,  0,  1
    };

    int Gy[9] = {
        -1, -2, -1,
         0,  0,  0,
         1,  2,  1
    };

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            for (int kr = -1; kr <= 1; kr++)
            {
                for (int kc = -1; kc <= 1; kc++, idx++)
                {
                    int x = clamp(j + kc, 0, width - 1);
                    int y = clamp(i + kr, 0, height - 1);
                    int pos = y * width + x;

                    rx += in_r[pos] * Gx[idx];
                    ry += in_r[pos] * Gy[idx];

                    gx += in_g[pos] * Gx[idx];
                    gy += in_g[pos] * Gy[idx];

                    bx += in_b[pos] * Gx[idx];
                    by += in_b[pos] * Gy[idx];
                }
            }

            int r = (int) sqrt((double)(rx * rx + ry * ry));
            int g = (int) sqrt((double)(gx * gx + gy * gy));
            int b = (int) sqrt((double)(bx * bx + by * by));

            out_r[i * width + j] = (unsigned char) clamp(r, 0, 255);
            out_g[i * width + j] = (unsigned char) clamp(g, 0, 255);
            out_b[i * width + j] = (unsigned char) clamp(b, 0, 255);

            rx = ry = gx = gy = bx = by = idx = 0;
        }
    }
}


/* sundiastiki odigia: target teams distribute parallel for collapse(2) */
void sobel_omp_device(img_t *imgin, img_t *imgout)
{
    int width  = imgin->header.width;
    int height = imgin->header.height;
    int total  = width * height;

    unsigned char *in_r  = imgin->red;
    unsigned char *in_g  = imgin->green;
    unsigned char *in_b  = imgin->blue;
    unsigned char *out_r = imgout->red;
    unsigned char *out_g = imgout->green;
    unsigned char *out_b = imgout->blue;

    int Gx[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
    int Gy[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
	
	/*Η combined έκδοση βάζει όλο το offloading σε μία οδηγία: target teams distribute parallel for collapse(2). Έτσι τα δύο loops της εικόνας ενώνονται και κάθε iteration 
	αντιστοιχεί σε ένα pixel. Τα input κανάλια και τα Sobel kernels στέλνονται στη GPU με map(to:), ενώ τα output κανάλια επιστρέφουν με map(from:).*/
	
	
    #pragma omp target teams distribute parallel for collapse(2) \ /*Το target μεταφέρει την εκτέλεση στη GPU.Επίσης Το collapse(2) ενώνει τα δύο loops
	γραμμές και στήλες, σε έναν μεγάλο χώρο επαναλήψεων. Έτσι ο compiler μοιράζει πιο εύκολα όλα τα pixels στα teams και threads.*/ 
        map(to: in_r[0:total], in_g[0:total], in_b[0:total]) \ /*Το map λέει ποια δεδομένα μεταφέρονται μεταξύ CPU και GPU.*/
        map(to: Gx[0:9], Gy[0:9]) \
        map(from: out_r[0:total], out_g[0:total], out_b[0:total]) \
        num_teams(NUM_TEAMS) thread_limit(THREAD_LIMIT) /* Το num_teams καθορίζει πόσες ομάδες δημιουργούνται στη GPU και το thread_limit καθορίζει πόσα threads μπορεί 
		να έχει κάθε ομάδα. Τα thread limits τα έβαλα πολλαπλάσια του 32 επειδή στις NVIDIA GPUs το warp έχει 32 threads.*/
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int rx = 0, ry = 0, gx = 0, gy = 0, bx = 0, by = 0; /*Οι μεταβλητές rx, ry, gx, gy, bx, by είναι τοπικές για κάθε iteration/pixel, άρα κάθε thread έχει δικά του 
																αθροίσματα και δεν υπάρχει race condition.*/

            for (int kr = -1; kr <= 1; kr++) {
                for (int kc = -1; kc <= 1; kc++) {
                    int idx = (kr + 1) * 3 + (kc + 1);
                    int x = j + kc;
                    int y = i + kr;

                    /* clamp inline (akrwn tis eikonas) */
                    if (x < 0) x = 0;
                    if (x >= width)  x = width - 1;
                    if (y < 0) y = 0;
                    if (y >= height) y = height - 1;

                    int pos = y * width + x;

                    rx += in_r[pos] * Gx[idx];
                    ry += in_r[pos] * Gy[idx];
                    gx += in_g[pos] * Gx[idx];
                    gy += in_g[pos] * Gy[idx];
                    bx += in_b[pos] * Gx[idx];
                    by += in_b[pos] * Gy[idx];
                }
            }

            int r = (int)sqrt((double)(rx * rx + ry * ry));
            int g = (int)sqrt((double)(gx * gx + gy * gy));
            int b = (int)sqrt((double)(bx * bx + by * by));

            /* clamp ston [0,255]. Den exoume kato apo 0 giati to sqrt vgazei thetiko. */
            if (r > 255) r = 255;
            if (g > 255) g = 255;
            if (b > 255) b = 255;

            out_r[i * width + j] = (unsigned char)r;
            out_g[i * width + j] = (unsigned char)g;
            out_b[i * width + j] = (unsigned char)b;
        }
    }
}

/* emfwleymenes odigies: teams distribute eksw, parallel for mesa.
 * stin arxi eixa kai edw collapse(2) alla evgaze idious xronous me tin
 * combined, opote to ebgala kai eyhike mikri sthatheri diafora (~10%) */
void sobel_omp_device_nested(img_t *imgin, img_t *imgout) /*Στη nested έκδοση έχω ξεχωριστές οδηγίες. Το target μεταφέρει την εκτέλεση στη GPU, το teams distribute 
										μοιράζει τον εξωτερικό βρόχο των γραμμών στα teams, και το parallel for μοιράζει τον εσωτερικό βρόχο των στηλών στα threads.*/
{
    int width  = imgin->header.width;
    int height = imgin->header.height;
    int total  = width * height;

    unsigned char *in_r  = imgin->red;
    unsigned char *in_g  = imgin->green;
    unsigned char *in_b  = imgin->blue;
    unsigned char *out_r = imgout->red;
    unsigned char *out_g = imgout->green;
    unsigned char *out_b = imgout->blue;

    int Gx[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
    int Gy[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };

    #pragma omp target map(to: in_r[0:total], in_g[0:total], in_b[0:total]) \
        map(to: Gx[0:9], Gy[0:9])                                            \
        map(from: out_r[0:total], out_g[0:total], out_b[0:total])
    {
        #pragma omp teams distribute num_teams(NUM_TEAMS) thread_limit(THREAD_LIMIT) /*Το teams δημιουργεί ομάδες από threads στη GPU
		κάτι αντίστοιχο με τα blocks στη CUDA και το distribute μοιράζει τις επαναλήψεις στα teams.*/
        for (int i = 0; i < height; i++) {
            #pragma omp parallel for
            for (int j = 0; j < width; j++) {
                int rx = 0, ry = 0, gx = 0, gy = 0, bx = 0, by = 0;

                for (int kr = -1; kr <= 1; kr++) {
                    for (int kc = -1; kc <= 1; kc++) {
                        int idx = (kr + 1) * 3 + (kc + 1);
                        int x = j + kc;
                        int y = i + kr;

                        if (x < 0) x = 0;
                        if (x >= width)  x = width - 1;
                        if (y < 0) y = 0;
                        if (y >= height) y = height - 1;

                        int pos = y * width + x;

                        rx += in_r[pos] * Gx[idx];
                        ry += in_r[pos] * Gy[idx];
                        gx += in_g[pos] * Gx[idx];
                        gy += in_g[pos] * Gy[idx];
                        bx += in_b[pos] * Gx[idx];
                        by += in_b[pos] * Gy[idx];
                    }
                }

                int r = (int)sqrt((double)(rx * rx + ry * ry));
                int g = (int)sqrt((double)(gx * gx + gy * gy));
                int b = (int)sqrt((double)(bx * bx + by * by));

                if (r > 255) r = 255;
                if (g > 255) g = 255;
                if (b > 255) b = 255;

                out_r[i * width + j] = (unsigned char)r;
                out_g[i * width + j] = (unsigned char)g;
                out_b[i * width + j] = (unsigned char)b;
            }
        }
    }
}

/* xronometrisi */
double get_time(void)
{
    return omp_get_wtime();
}

/* tsek oti pragmatika trexei sti GPU */
void print_offloading_status(void)
{
    int num_devs = omp_get_num_devices();
    int def_dev = omp_get_default_device();
    int in_target_says_initial = -1;

    #pragma omp target map(tofrom: in_target_says_initial)
    {
        in_target_says_initial = omp_is_initial_device();
    }

    printf("OpenMP offloading status:\n");
    printf("  omp_get_num_devices()     = %d\n", num_devs);
    printf("  omp_get_default_device()  = %d\n", def_dev);
    printf("  in-target omp_is_initial_device() = %d  -> %s\n",
           in_target_says_initial,
           in_target_says_initial == 0 ? "running on device (correct)" : "running on host/fallback");
    printf("  GPU offloading available: %s (devices=%d)\n\n",
           num_devs > 0 ? "YES" : "NO", num_devs);
}

/* trexei mia synartisi RUNS fores kai gyrnaei to meso oro.
 * exei kai mia warm-up ektelesh prin gia na zestathei i GPU */
double timeit_n(void (*f)(img_t *, img_t *), img_t *imgin, img_t *imgout,
                int runs, const char *label)
{
    double total = 0.0;

    /* Warm-up: trexei kanonika ti synartisi alla o xronos den metraei. */
    double tw0 = get_time();
    f(imgin, imgout);
    double tw1 = get_time();
    printf("%s warm-up: %.6f sec (not counted)\n", label, tw1 - tw0);

    for (int r = 0; r < runs; r++) {
        double t0 = get_time();
        f(imgin, imgout);
        double t1 = get_time();
        printf("%s run %d: %.6f sec\n", label, r + 1, t1 - t0);
        total += t1 - t0;
    }

    return total / runs;
}

/* afairei tin epektasi tou arxeiou (px "1500.bmp" -> "1500") */
char *remove_ext(char *str, char extsep, char pathsep)
{
    char *newstr, *ext, *lpath;

    if (str == NULL) return NULL;
    if ((newstr = malloc(strlen(str) + 1)) == NULL) return NULL;

    strcpy(newstr, str);
    ext = strrchr(newstr, extsep);
    lpath = (pathsep == 0) ? NULL : strrchr(newstr, pathsep);
    if (ext != NULL)
    {
        if (lpath != NULL)
        {
            if (lpath < ext)
                *ext = '\0';
        }
        else
            *ext = '\0';
    }
    return newstr;
}

/* sygkrisi pixel-pixel, gyrnaei posa diaferoun */
int compare_images(img_t *a, img_t *b)
{
    int total  = a->header.width * a->header.height;
    int errors = 0;

    for (int i = 0; i < total; i++) {
        if (a->red[i] != b->red[i] ||
            a->green[i] != b->green[i] ||
            a->blue[i] != b->blue[i]) {
            errors++;
            if (errors <= 5)
                printf("  pixel %d differs\n", i);
        }
    }

    return errors;
}

int main(int argc, char *argv[])
{
    img_t imgin;
    img_t imgout_serial;
    img_t imgout_gpu;
    img_t imgout_nested;

    char *inputfile, *noextfname;
    char serial_file[128], gpu_file[128], nested_file[128];
    int runs = 4;       /* 4 ektelseis kai meso oro */

    if (argc < 2) {
        fprintf(stderr, "Syntax: %s <filename>, e.g. %s 1500.bmp\n",
                argv[0], argv[0]);
        fprintf(stderr, "Available images: 500.bmp, 1000.bmp, 1500.bmp\n");
        return 1;
    }

    inputfile  = argv[1];
    noextfname = remove_ext(inputfile, '.', '/');
    if (noextfname == NULL) {
        fprintf(stderr, "Cannot allocate memory for output filename; exiting.\n");
        return 1;
    }

    if (snprintf(serial_file, sizeof(serial_file), "%s-serial.bmp", noextfname) >= (int)sizeof(serial_file) ||
        snprintf(gpu_file, sizeof(gpu_file), "%s-omp-device.bmp", noextfname) >= (int)sizeof(gpu_file) ||
        snprintf(nested_file, sizeof(nested_file), "%s-omp-nested.bmp", noextfname) >= (int)sizeof(nested_file)) {
        fprintf(stderr, "Output filename is too long; exiting.\n");
        free(noextfname);
        return 1;
    }

    /* diavasma eikonas kai allocations */
    bmp_read_img_from_file(inputfile, &imgin);
    bmp_clone_empty_img(&imgin, &imgout_serial);
    bmp_clone_empty_img(&imgin, &imgout_gpu);
    bmp_clone_empty_img(&imgin, &imgout_nested);

    bmp_rgb_alloc(&imgin);
    bmp_rgb_alloc(&imgout_serial);
    bmp_rgb_alloc(&imgout_gpu);
    bmp_rgb_alloc(&imgout_nested);

    bmp_rgb_from_data(&imgin);

    printf("<<< Sobel filter (h=%d, w=%d) >>>\n",
           imgin.header.height, imgin.header.width);
    printf("NUM_TEAMS=%d THREAD_LIMIT=%d, runs=%d (+1 warm-up)\n",
           NUM_TEAMS, THREAD_LIMIT, runs);

    print_offloading_status();

    /* trexoume kai tis 3 ekdoseis */
    double serial_avg = timeit_n(sobel_serial, &imgin, &imgout_serial, runs, "serial");
    printf("serial avg: %.6f sec\n\n", serial_avg);

    double gpu_avg = timeit_n(sobel_omp_device, &imgin, &imgout_gpu, runs, "gpu-combined");
    printf("gpu (combined) avg: %.6f sec\n\n", gpu_avg);

    double nested_avg = timeit_n(sobel_omp_device_nested, &imgin, &imgout_nested, runs, "gpu-nested");
    printf("gpu (nested) avg: %.6f sec\n\n", nested_avg);

    /* epalitheysi me ti seiriaki */
    int err1 = compare_images(&imgout_serial, &imgout_gpu);
    int err2 = compare_images(&imgout_serial, &imgout_nested);

    printf("Results:\n");
    printf("  serial:       %.6f sec\n", serial_avg);
    printf("  gpu combined: %.6f sec  speedup=%.2fx\n", gpu_avg,    serial_avg / gpu_avg);
    printf("  gpu nested:   %.6f sec  speedup=%.2fx\n", nested_avg, serial_avg / nested_avg);
    printf("verification combined: %s\n", err1 == 0 ? "OK" : "WRONG");
    printf("verification nested:   %s\n", err2 == 0 ? "OK" : "WRONG");

    /* save twn 3 ekdosewn ws BMP gia na vlepoume kai me to mati */
    bmp_data_from_rgb(&imgout_serial);
    bmp_data_from_rgb(&imgout_gpu);
    bmp_data_from_rgb(&imgout_nested);

    bmp_write_data_to_file(serial_file,  &imgout_serial);
    bmp_write_data_to_file(gpu_file,     &imgout_gpu);
    bmp_write_data_to_file(nested_file,  &imgout_nested);

    bmp_img_free(&imgin);
    bmp_img_free(&imgout_serial);
    bmp_img_free(&imgout_gpu);
    bmp_img_free(&imgout_nested);

    free(noextfname);

    return 0;
}
