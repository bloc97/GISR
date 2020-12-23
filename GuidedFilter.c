/*------------------------------------------------------*/
/* Prog    : GuidedFilter.c                             */
/* Auteur  : Bowen Peng                                 */
/* Date    :                                            */
/* version :                                            */ 
/* langage : C                                          */
/* labo    : DIRO                                       */
/*------------------------------------------------------*/

/*------------------------------------------------*/
/* FICHIERS INCLUS -------------------------------*/
/*------------------------------------------------*/
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*------------------------------------------------*/
/* DEFINITIONS -----------------------------------*/
/*------------------------------------------------*/
#define NAME_IMG_IN  "texture"

#define NAME_IMG_OUT "ImgSegmented"

#define RAND_SEED time(0)

#define SWAP(a,b) tempr=(a);(a)=(b);(b)=tempr
#define CARRE(X) ((X)*(X))
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))


#define NBCHAR 200

#define PI 3.141592654
/*------------------------------------------------*/
/* FONCTIONS -------------------------------------*/                     
/*------------------------------------------------*/

/*---------------------------------------------------------*/
/*  Alloue de la memoire pour une matrice 1d de float      */
/*---------------------------------------------------------*/
float* fmatrix_allocate(int size) {
	float* matrix = (float*) malloc(sizeof(float)*size); 
	if (matrix==NULL) printf("probleme d'allocation memoire");
	return matrix; 
}

/*----------------------------------------------------------*/
/* Chargement de l'image de nom <name> (en pgm)             */
/*----------------------------------------------------------*/
float* LoadImagePgm(char* name,int *length,int *width) {
	unsigned char var;
	char buff[NBCHAR];
	float* mat;

	char stringTmp1[NBCHAR];

	int ta1,ta2,ta3;
	FILE *fic;

	/*-----nom du fichier pgm-----*/
	strcpy(buff,name);
	strcat(buff,".pgm");
	printf("---> Ouverture de %s\n", buff);

	/*----ouverture du fichier----*/
	fic=fopen(buff,"r");
	if (fic==NULL)
	{ printf("\n- Grave erreur a l'ouverture de %s  -\n",buff);
	  exit(-1); }

	/*--recuperation de l'entete--*/
	fgets(stringTmp1,NBCHAR-1,fic);
	fread(&var,1,1,fic);
	fseek(fic,-1,SEEK_CUR);
	if (var == '#') {
	  fgets(stringTmp1,NBCHAR-1,fic);
	}
	fscanf(fic,"%d %d",&ta1,&ta2);
	fscanf(fic,"%d",&ta3);
	fgets(stringTmp1,NBCHAR-1,fic);

	*width=ta1;
	*length=ta2;
	mat=fmatrix_allocate(ta1*ta2);

	/*--chargement dans la matrice--*/
	 for(int i=0;i<ta2;i++)
	  for(int j=0;j<ta1;j++)  
		{ fread(&var,1,1,fic);
		  mat[i*ta1+j]=var/255.0f; }

	/*---fermeture du fichier---*/
	fclose(fic);

	return mat;
}

/*----------------------------------------------------------*/
/* Sauvegarde de l'image de nom <name> au format pgm        */
/*----------------------------------------------------------*/
void SaveImagePgm(char* name, float* mat, int length, int width) {
  int i,j,k;
  char buff[NBCHAR];
  FILE* fic;
  time_t tm;

  /*--extension--*/
  strcpy(buff,name);
  strcat(buff,".pgm");

  /*--ouverture fichier--*/
  fic=fopen(buff,"w");
    if (fic==NULL) 
        { printf(" Probleme dans la sauvegarde de %s",buff); 
          exit(-1); }
	printf("<--- Sauvegarde de %s\n", buff);

  /*--sauvegarde de l'entete--*/
  fprintf(fic,"P5");
  if (ctime(&tm)==NULL) fprintf(fic,"\n#\n");
  else fprintf(fic,"\n# IMG Module, %s",ctime(&tm));
  fprintf(fic,"%d %d",width,length);
  fprintf(fic,"\n255\n");

  /*--enregistrement--*/
     for(i=0;i<length*width;i++) {
		 float val = mat[i] * 255.0f;
		 if (val < 0) {
			 val = 0;
		 } else if (val > 255) {
			 val = 255;
		 }
		 
        fprintf(fic,"%c",(char)val);
	 }
   
  /*--fermeture fichier--*/
   fclose(fic); 
 } 

//Edge repeat safe sampling (prevents array out of bounds)
float SafeSample(float* m, int x, int y, int w, int h) {
	x = (x < 0) ? 0 : ((x >= w) ? (w - 1) : x);
	y = (y < 0) ? 0 : ((y >= h) ? (h - 1) : y);
	return m[y*w+x];
}

//Linear time O(h*w) mean filter
void FastMean(float* output, float* image, int height, int width, int kernel_size) {
	int k2 = kernel_size / 2;
	int k2i = k2 + kernel_size % 2;
	float temp[height*width];
	
	for (int i=0; i<height; i++) {
		float val = 0;
		for (int j=-k2i; j<k2; j++) {
			val += SafeSample(image, j, i, width, height);
		}
		
		for (int j=0; j<width; j++) {
			val += SafeSample(image, j + k2, i, width, height);
			val -= SafeSample(image, j - k2i, i, width, height);
			temp[width*i+j] = val / kernel_size;
		}
	}
	
	for (int j=0; j<width; j++) {
		float val = 0;
		for (int i=-k2i; i<k2; i++) {
			val += SafeSample(temp, j, i, width, height);
		}
		
		for (int i=0; i<height; i++) {
			val += SafeSample(temp, j, i + k2, width, height);
			val -= SafeSample(temp, j, i - k2i, width, height);
			output[width*i+j] = val / kernel_size;
		}
	}
}

void ElementAddScalar(float* output, float* in0, float s, int size) {
	for (int i=0; i<size; i++) {
		output[i] = in0[i] + s;
	}
}

void ElementAdd(float* output, float* in0, float* in1, int size) {
	for (int i=0; i<size; i++) {
		output[i] = in0[i] + in1[i];
	}
}

void ElementSub(float* output, float* in0, float* in1, int size) {
	for (int i=0; i<size; i++) {
		output[i] = in0[i] - in1[i];
	}
}

void ElementMul(float* output, float* in0, float* in1, int size) {
	for (int i=0; i<size; i++) {
		output[i] = in0[i] * in1[i];
	}
}

void ElementDiv(float* output, float* in0, float* in1, int size) {
	for (int i=0; i<size; i++) {
		output[i] = in0[i] / in1[i];
	}
}

//Bilinear interpolation
float SampleBilinear(float* m, float x, float y, int w, int h) {
	int x0 = (int)floorf(x);
	int x1 = (int)ceilf(x);
	int y0 = (int)floorf(y);
	int y1 = (int)ceilf(y);
	
	float ax1 = x - x0;
	float ay1 = y - y0;
	float ax0 = 1.0f - ax1;
	float ay0 = 1.0f - ay1;
	
	float x0y0 = ax0 * ay0;
	float x1y0 = ax1 * ay0;
	float x0y1 = ax0 * ay1;
	float x1y1 = ax1 * ay1;
	
	return SafeSample(m, x0, y0, w, h) * x0y0 
		 + SafeSample(m, x1, y0, w, h) * x1y0 
		 + SafeSample(m, x0, y1, w, h) * x0y1 
		 + SafeSample(m, x1, y1, w, h) * x1y1;
}

//Resamples image using bilinear interpolation
void Resample(float* output, int target_height, int target_width, float* image, int height, int width) {
	for (int i=0; i<(target_height*target_width); i++) {
		int x = i % target_width;
		int y = i / target_width;
		
		float xr = (float)x / (float)target_width * (float)width;
		float yr = (float)y / (float)target_height * (float)height;
		
		output[i] = SampleBilinear(image, xr, yr, width, height);
	}
}

//Resampling that takes in account the sampling theorem, reduces aliasing when downsampling factor is greater than 2
void ResampleNyquist(float* output, int target_height, int target_width, float* image, int height, int width) {
	float yfactor = (float)target_height / (float)height;
	float xfactor = (float)target_width / (float)width;
	
	if (xfactor >= 0.5 && yfactor >= 0.5) {
		Resample(output, target_height, target_width, image, height, width);
	} else {
		int newheight = MAX(height/2 + (height%2), target_height);
		int newwidth = MAX(width/2 + (width%2), target_width);
		float newyfactor = (float)newheight / (float)height;
		float newxfactor = (float)newwidth / (float)height;
		
		if (newyfactor < 0.5) {
			newheight += 1;
		}
		if (newxfactor < 0.5) {
			newwidth += 1;
		}
		
		float* temp = fmatrix_allocate(newheight*newwidth);
		Resample(temp, newheight, newwidth, image, height, width);
		ResampleNyquist(output, target_height, target_width, temp, newheight, newwidth);
		free(temp);
	}
}

//Compute the guided filter
void GuidedFilter(float* output, float* image, float* guide_ori, int height, int width, int height_guide, int width_guide, int kernel_size, float epsilon) {
	int size = height * width;
	float* temp = fmatrix_allocate(size);
	float* meanI = fmatrix_allocate(size);
	float* meanP = fmatrix_allocate(size);
	
	//Resize the guide so it is the same size as the image
	float* guide = fmatrix_allocate(size);
	ResampleNyquist(guide, height, width, guide_ori, height_guide, width_guide);
	
	//Compute statistics
	FastMean(meanI, guide, height, width, kernel_size);
	FastMean(meanP, image, height, width, kernel_size);
	
	float* corrI = fmatrix_allocate(size);
	float* corrIP = fmatrix_allocate(size);
	
	ElementMul(temp, guide, guide, size);
	FastMean(corrI, temp, height, width, kernel_size);
	
	ElementMul(temp, guide, image, size);
	FastMean(corrIP, temp, height, width, kernel_size);
	free(guide);
	
	float* varI = fmatrix_allocate(size);
	float* covIP = fmatrix_allocate(size);
	
	ElementMul(temp, meanI, meanI, size);
	ElementSub(varI, corrI, temp, size);
	
	ElementMul(temp, meanI, meanP, size);
	ElementSub(covIP, corrIP, temp, size);
	free(corrI);
	free(corrIP);
	
	//Compute a and b
	float* a = fmatrix_allocate(size);
	float* b = fmatrix_allocate(size);
	
	ElementAddScalar(temp, varI, epsilon, size);
	ElementDiv(a, covIP, temp, size);
	
	ElementMul(temp, a, meanI, size);
	ElementSub(b, meanP, temp, size);
	free(meanI);
	free(meanP);
	free(varI);
	free(covIP);
	free(temp);
	
	FastMean(a, a, height, width, kernel_size);
	FastMean(b, b, height, width, kernel_size);
	
	float* am = fmatrix_allocate(height_guide * width_guide);
	float* bm = fmatrix_allocate(height_guide * width_guide);
	
	//Resize a and b to be the same size as the guide
	ResampleNyquist(am, height_guide, width_guide, a, height, width);
	ResampleNyquist(bm, height_guide, width_guide, b, height, width);
	free(a);
	free(b);
	
	ElementMul(am, am, guide_ori, height_guide * width_guide);
	ElementAdd(output, am, bm, height_guide * width_guide);
	
	free(am);
	free(bm);
}


/*------------------------------------------------*/
/* PROGRAMME PRINCIPAL ---------------------------*/                     
/*------------------------------------------------*/
int main(int argc, char *argv[]) {
	
 	if(argc<6){
		printf("Usage :\n\t GuidedFilter image image_guide image_output kernel_size epsilon\n\n");
		return 0;
	}
	
	//Memory allocation
	int height,width;
	int height_guide,width_guide;
	float* InputImage = LoadImagePgm(argv[1],&height,&width);
	float* InputGuide = LoadImagePgm(argv[2],&height_guide,&width_guide);
	
	float* DownscaledGuide = fmatrix_allocate(height*width);
	float* DownscaledMatA = fmatrix_allocate(height*width);
	float* DownscaledMatB = fmatrix_allocate(height*width);
	float* MatA = fmatrix_allocate(height_guide*width_guide);
	float* MatB = fmatrix_allocate(height_guide*width_guide);
	float* OutputImage = fmatrix_allocate(height_guide*width_guide);
	
	int kernel_size = atoi(argv[4]);
	float epsilon = atof(argv[5]);
	
	if (epsilon < 1e-3f) {
		epsilon = 1e-3f;
	}
	
	GuidedFilter(OutputImage, InputImage, InputGuide, height, width, height_guide, width_guide, kernel_size, epsilon * epsilon);
	
	/*-------- FIN ---------------------------------------------*/
	/*----------------------------------------------------------*/
	/*Sauvegarde des matrices sous forme d'image pgms*/
	SaveImagePgm(argv[3],OutputImage,height_guide,width_guide);

	/*Liberation memoire pour les matrices*/
	free(InputImage);
	free(InputGuide);
	free(DownscaledGuide);
	free(DownscaledMatA);
	free(DownscaledMatB);
	free(MatA);
	free(MatB);
	free(OutputImage);

	/*retour sans probleme*/ 
	printf("\n");
	return 0;
}
