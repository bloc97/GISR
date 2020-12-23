/*------------------------------------------------------*/
/* Prog    : GuidedFilter_Color.c                       */
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
#define RAND_SEED time(0)

#define SWAP(a,b) tempr=(a);(a)=(b);(b)=tempr
#define CARRE(X) ((X)*(X))
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#define NBCHAR 200
/*------------------------------------------------*/
/* FONCTIONS -------------------------------------*/                     
/*------------------------------------------------*/

/*---------------------------------------------------------*/
/*  Alloue de la memoire pour une matrice 1d de float      */
/*---------------------------------------------------------*/
float* fmatrix_allocate(int size) {
	float* matrix = (float*)malloc(sizeof(float)*size); 
	if (matrix==NULL) printf("probleme d'allocation memoire");
	return matrix; 
}

/*----------------------------------------------------------*/
/* Chargement de l'image de nom <name> (en ppm)             */
/*----------------------------------------------------------*/
float* LoadImagePpm(char* name,int *length,int *width) {
	unsigned char var;
	char buff[NBCHAR];
	float* mat;

	char stringTmp1[NBCHAR];

	int ta1,ta2,ta3;
	FILE *fic;

	/*-----nom du fichier pgm-----*/
	strcpy(buff,name);
	strcat(buff,".ppm");
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
	int size = ta1*ta2*3;
	mat=fmatrix_allocate(size);

	/*--chargement dans la matrice--*/
	 for(int i=0; i<size; i++) {
		fread(&var,1,1,fic);
		mat[i] = (float)var/255.0f;
	}

	/*---fermeture du fichier---*/
	fclose(fic);

	return mat;
}

float clamprgb(float x) {
	if (x < 0) {
		x = 0;
	} else if (x > 255) {
		x = 255;
	}
	return x;
}

/*----------------------------------------------------------*/
/* Sauvegarde de l'image de nom <name> au format ppm        */
/*----------------------------------------------------------*/
void SaveImagePpm(char* name, float* mat, int length, int width) {
	char buff[NBCHAR];
	FILE* fic;
	time_t tm;

	/*--extension--*/
	strcpy(buff,name);
	strcat(buff,".ppm");

	/*--ouverture fichier--*/
	fic=fopen(buff,"w");
	if (fic==NULL) 
		{ printf(" Probleme dans la sauvegarde de %s",buff); 
		  exit(-1); }
	printf("<--- Sauvegarde de %s\n", buff);

	/*--sauvegarde de l'entete--*/
	fprintf(fic,"P6");
	if (ctime(&tm)==NULL) fprintf(fic,"\n#\n");
	else fprintf(fic,"\n# IMG Module, %s",ctime(&tm));
	fprintf(fic,"%d %d",width,length);
	fprintf(fic,"\n255\n");

	/*--enregistrement--*/
	int size = length*width*3;
	for(int i=0; i<size; i++) {
		float val = clamprgb(mat[i] * 255.0f);
		fprintf(fic,"%c",(char)val);
	}

	/*--fermeture fichier--*/
	fclose(fic); 
}

//Edge repeat safe sampling (prevents array out of bounds)
float SafeSample(float* m, int c, int x, int y, int d, int w, int h) {
	x = (x < 0) ? 0 : ((x >= w) ? (w - 1) : x);
	y = (y < 0) ? 0 : ((y >= h) ? (h - 1) : y);
	return m[y*w*d+x*d+c];
}

//Linear time O(h*w*d) mean filter
void FastMean(float* output, float* image, int height, int width, int depth, int kernel_size) {
	int k2 = kernel_size / 2;
	int k2i = k2 + kernel_size % 2;
	float* temp = fmatrix_allocate(height*width*depth);
	
	for (int c=0; c<depth; c++) {
		for (int i=0; i<height; i++) {
			float val = 0;
			for (int j=-k2i; j<k2; j++) {
				val += SafeSample(image, c, j, i, depth, width, height);
			}
			
			for (int j=0; j<width; j++) {
				val += SafeSample(image, c, j + k2, i, depth, width, height);
				val -= SafeSample(image, c, j - k2i, i, depth, width, height);
				temp[width*depth*i+depth*j+c] = val / kernel_size;
			}
		}
		
		for (int j=0; j<width; j++) {
			float val = 0;
			for (int i=-k2i; i<k2; i++) {
				val += SafeSample(temp, c, j, i, depth, width, height);
			}
			
			for (int i=0; i<height; i++) {
				val += SafeSample(temp, c, j, i + k2, depth, width, height);
				val -= SafeSample(temp, c, j, i - k2i, depth, width, height);
				output[width*depth*i+depth*j+c] = val / kernel_size;
			}
		}
	}
	
	free(temp);
}

void ElementCopy(float* output, float* in0, int size) {
	for (int i=0; i<size; i++) {
		output[i] = in0[i];
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

//Invert huge l*3x3 symmetric covariance tensor
void Invert3x3CovarianceTensor(float* m, int size) {
	int depthcorr = 6;
	for (int i=0; i<size; i++) {
		int idx = i * depthcorr;
		
		int i00 = idx+0;
		int i01 = idx+1;
		int i02 = idx+2;
		int i11 = idx+3;
		int i12 = idx+4;
		int i22 = idx+5;
		

		double a00 = (double)m[i11] * m[i22] - (double)m[i12] * m[i12];
		double a01 = (double)m[i02] * m[i12] - (double)m[i01] * m[i22];
		double a02 = (double)m[i01] * m[i12] - (double)m[i02] * m[i11];
		double a11 = (double)m[i00] * m[i22] - (double)m[i02] * m[i02];
		double a12 = (double)m[i01] * m[i02] - (double)m[i00] * m[i12];
		double a22 = (double)m[i00] * m[i11] - (double)m[i01] * m[i01];
		
		double a01i = (double)m[i01] * m[i22] - (double)m[i12] * m[i02];
		
		double det = ((double)m[i00] * a00) - ((double)m[i01] * a01i) + ((double)m[i02] * a02);
		
		a00 /= det;
		a01 /= det;
		a02 /= det;
		a11 /= det;
		a12 /= det;
		a22 /= det;
		
		m[idx + 0] = a00;
		m[idx + 1] = a01;
		m[idx + 2] = a02;
		m[idx + 3] = a11;
		m[idx + 4] = a12;
		m[idx + 5] = a22;
	}
}

//Get index of symmetric covariance matrix
int GetTriangularMatrixIndex(int j, int i, int size) {
	
	if (j > i) {
		int tmp = j;
		j = i;
		i = tmp;
	}
	
	int removedindices = (j * (j+1)) / 2;
	
	
	return j * size + i - removedindices;
}

//Dot product of tensors
void TensorDotProduct(float* out, float* u, float* v, int size, int depth) {
	for (int n=0; n<size; n++) {
		int idx = n * depth * depth;
		int cidx = n * ((depth * (depth+1)) / 2);
		for (int i=0; i<depth; i++) {
			for (int j=0; j<depth; j++) {
				out[idx + i*depth + j] = 0;
				for (int k=0; k<depth; k++) {
					out[idx + i*depth + j] += u[idx + k*depth + i] * v[cidx + GetTriangularMatrixIndex(j, k, depth)];
				}
			}
		}
	}
}
//Dot product of vectors
void VectorDotProduct(float* out, float* m, float* v, int size, int depth) {
	for (int n=0; n<size; n++) {
		int idx = n * depth;
		for (int i=0; i<depth; i++) {
			out[idx + i] = 0;
			for (int k=0; k<depth; k++) {
				out[idx + i] += m[idx*depth + i*depth + k] * v[idx + k];
			}
		}
	}
}

//Add diagonal identity matrix multiplied with epsilon to covariance matrix
void AddDiagonalToCorrelation(float* m, int size, int depth, int depthcorr, float epsilon) {
	for (int k=0; k<size; k++) {
		int knd = k * depthcorr;
		
		int corridx = 0;
		for (int i=0; i<depth; i++) {
			m[knd + GetTriangularMatrixIndex(i, i, depth)] += epsilon;
		}
	}
}

//Transpose of matrix
void Transpose(float* m, int size, int depth) {
	for (int k=0; k<size; k++) {
		int knd = k * depth * depth;
		
		for (int i=0; i<depth; i++) {
			for (int j=i+1; j<depth; j++) {
				int ij = knd + i*depth + j;
				int ji = knd + j*depth + i;
				
				float tmp = m[ij];
				m[ij] = m[ji];
				m[ji] = tmp;
			}
		}
	}
}

//Compute correlation matrix between all pairs of values in m
void ComputeCorrelation(float* out, float* m, int size, int depth, int depthcorr, int height, int width, int kernel_size) {
	for (int k=0; k<size; k++) {
		int knd = k * depthcorr;
		int kd = k * depth;
		
		int corridx = 0;
		for (int i=0; i<depth; i++) {
			for (int j=i; j<depth; j++) {
				out[knd + corridx] = m[kd + i] * m[kd + j];
				corridx++;
			}
		}
	}
}

//Compute cross-correlation matrix between all pairs of values in m0 and m1
void ComputeCrossCorrelation(float* out, float* m0, float* m1, int size, int depth, int depthcross, int height, int width, int kernel_size) {
	for (int k=0; k<size; k++) {
		int knd = k * depthcross;
		int kd = k * depth;
		
		int corridx = 0;
		for (int i=0; i<depth; i++) {
			for (int j=0; j<depth; j++) {
				out[knd + corridx] = m0[kd + i] * m1[kd + j];
				corridx++;
			}
		}
	}
}

//Bilinear interpolation
float SampleBilinear(float* m, float c, float x, float y, int d, int w, int h) {
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
	
	return SafeSample(m, c, x0, y0, d, w, h) * x0y0 
		 + SafeSample(m, c, x1, y0, d, w, h) * x1y0 
		 + SafeSample(m, c, x0, y1, d, w, h) * x0y1 
		 + SafeSample(m, c, x1, y1, d, w, h) * x1y1;
}

//Resamples image using bilinear interpolation
void Resample(float* output, int target_height, int target_width, float* image, int height, int width, int depth) {
	for (int i=0; i<(target_height*target_width); i++) {
		int x = i % target_width;
		int y = i / target_width;
		
		float xr = (float)x / (float)target_width * (float)width;
		float yr = (float)y / (float)target_height * (float)height;
		for (int j=0; j<depth; j++) {
			output[i*depth+j] = SampleBilinear(image, j, xr, yr, depth, width, height);
		}
	}
}

//Resampling that takes in account the sampling theorem, reduces aliasing when downsampling factor is greater than 2
void ResampleNyquist(float* output, int target_height, int target_width, float* image, int height, int width, int depth) {
	float yfactor = (float)target_height / (float)height;
	float xfactor = (float)target_width / (float)width;
	
	if (xfactor >= 0.5 && yfactor >= 0.5) {
		Resample(output, target_height, target_width, image, height, width, depth);
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
		
		float* temp = fmatrix_allocate(newheight*newwidth*depth);
		Resample(temp, newheight, newwidth, image, height, width, depth);
		ResampleNyquist(output, target_height, target_width, temp, newheight, newwidth, depth);
		free(temp);
	}
}



//Compute the guided filter
void GuidedFilter(float* output_rgb, float* image_rgb, float* guide_rgb_ori, int height, int width, int height_guide, int width_guide, int depth, int kernel_size, float epsilon) {
	int depthcross = depth * depth;
	int depthcorr = (depth * (depth + 1)) / 2;
	int size = height * width;
	int sizedepth = size * depth;
	int sizedepthcross = size * depthcross;
	int sizedepthcorr = size * depthcorr;
	
	float* guide_rgb = fmatrix_allocate(sizedepth);
	ResampleNyquist(guide_rgb, height, width, guide_rgb_ori, height_guide, width_guide, depth);
	
	float* meanP = fmatrix_allocate(sizedepth);
	FastMean(meanP, image_rgb, height, width, depth, kernel_size);
	
	float* meanI = fmatrix_allocate(sizedepth);
	FastMean(meanI, guide_rgb, height, width, depth, kernel_size);
	
	float* corrI = fmatrix_allocate(sizedepthcorr);
	float* covI = fmatrix_allocate(sizedepthcorr);
	ComputeCorrelation(corrI, guide_rgb, size, depth, depthcorr, height, width, kernel_size);
	FastMean(corrI, corrI, height, width, depthcorr, kernel_size);
	ComputeCorrelation(covI, meanI, size, depth, depthcorr, height, width, kernel_size);
	ElementSub(covI, corrI, covI, sizedepthcorr);
	free(corrI);
	
	AddDiagonalToCorrelation(covI, size, depth, depthcorr, epsilon);
	
	//Invert covariance tensor
	Invert3x3CovarianceTensor(covI, size);
	
	float* corrIP = fmatrix_allocate(sizedepthcross);
	float* covIP = fmatrix_allocate(sizedepthcross);
	ComputeCrossCorrelation(corrIP, guide_rgb, image_rgb, size, depth, depthcross, height, width, kernel_size);
	FastMean(corrIP, corrIP, height, width, depthcross, kernel_size);
	ComputeCrossCorrelation(covIP, meanI, meanP, size, depth, depthcross, height, width, kernel_size);
	ElementSub(covIP, corrIP, covIP, sizedepthcross);
	free(corrIP);
	free(guide_rgb);
	
	float* a = fmatrix_allocate(sizedepthcross);
	float* b = fmatrix_allocate(sizedepth);
	
	//Tensor products
	TensorDotProduct(a, covIP, covI, size, depth);
	free(covIP);
	free(covI);
	
	VectorDotProduct(b, a, meanI, size, depth);
	ElementSub(b, meanP, b, sizedepth);
	free(meanI);
	free(meanP);
	
	FastMean(a, a, height, width, depthcross, kernel_size);
	FastMean(b, b, height, width, depth, kernel_size);
	
	float* am = fmatrix_allocate(height_guide * width_guide * depthcross);
	float* bm = fmatrix_allocate(height_guide * width_guide * depth);
	
	ResampleNyquist(am, height_guide, width_guide, a, height, width, depthcross);
	ResampleNyquist(bm, height_guide, width_guide, b, height, width, depth);
	
	VectorDotProduct(output_rgb, am, guide_rgb_ori, height_guide * width_guide, depth);
	ElementAdd(output_rgb, output_rgb, bm, height_guide * width_guide * depth);
	
	free(a);
	free(b);
	free(am);
	free(bm);
}

//Colorspace conversions
void RGBToYUV(float* out, float* in, int size) {
	for (int i=0; i<size; i++) {
		int idx = i * 3;
		
		out[idx    ] = in[idx] *  0.299f   + in[idx + 1] *  0.587f   + in[idx + 2] *  0.114f  ;
		out[idx + 1] = in[idx] * -0.14713f + in[idx + 1] * -0.28886f + in[idx + 2] *  0.436f  ;
		out[idx + 2] = in[idx] *  0.615f   + in[idx + 1] * -0.51499f + in[idx + 2] * -0.10001f;
		
	}
}
void YUVtoRGB(float* out, float* in, int size) {
	for (int i=0; i<size; i++) {
		int idx = i * 3;
		
		out[idx    ] = in[idx]                /*0.0f*/   + in[idx + 2] *  1.13983f  ;
		out[idx + 1] = in[idx] + in[idx + 1] * -0.39465f + in[idx + 2] * -0.58060f  ;
		out[idx + 2] = in[idx] + in[idx + 1] *  2.03211f                /*0.0f*/    ;
		
	}
}

//Replace UV channels
void ReplaceUV(float* rep, float* in, int size) {
	for (int i=0; i<size; i++) {
		int idx = i * 3;
		rep[idx + 1] = in[idx + 1];
		rep[idx + 2] = in[idx + 2];
	}
}

//Convert RGB to YUV, replace UV and convert back to RGB
void ReplaceColor(float* out, float* in_luminance, float* in_color, int size) {
	float* yuv_y = fmatrix_allocate(size*3);
	float* yuv_uv = fmatrix_allocate(size*3);
	
	RGBToYUV(yuv_y, in_luminance, size);
	RGBToYUV(yuv_uv, in_color, size);
	ReplaceUV(yuv_y, yuv_uv, size);
	
	YUVtoRGB(out, yuv_y, size);
	
	free(yuv_y);
	free(yuv_uv);
}


/*------------------------------------------------*/
/* PROGRAMME PRINCIPAL ---------------------------*/                     
/*------------------------------------------------*/
int main(int argc, char *argv[]) {
	
 	if(argc<6){
		printf("Usage :\n\t GuidedFilter image image_guide image_output kernel_size epsilon (colorize)\n\n");
		return 0;
	}
	int doColorize = 0;
	if (argc >= 7) {
		doColorize = 1;
	}
	
	//Memory allocation
	int height,width;
	int height_guide,width_guide;
	float* InputImage = LoadImagePpm(argv[1], &height, &width);
	float* InputGuide = LoadImagePpm(argv[2], &height_guide, &width_guide);
	
	int depth = 3;
	float* OutputImage = fmatrix_allocate(height_guide*width_guide*depth);
	
	int kernel_size = atoi(argv[4]);
	float epsilon = atof(argv[5]);
	
	if (epsilon < 1e-3f) {
		epsilon = 1e-3f;
	}
	
	GuidedFilter(OutputImage, InputImage, InputGuide, height, width, height_guide, width_guide, depth, kernel_size, epsilon * epsilon * depth);
	
	if (doColorize) {
		ReplaceColor(OutputImage, InputGuide, OutputImage, height_guide*width_guide);
	}
	
	/*-------- FIN ---------------------------------------------*/
	/*----------------------------------------------------------*/
	/*Sauvegarde des matrices sous forme d'image pgms*/
	SaveImagePpm(argv[3], OutputImage, height_guide, width_guide);

	/*Liberation memoire pour les matrices*/
	free(InputImage);
	free(InputGuide);
	free(OutputImage);

	/*retour sans probleme*/ 
	printf("\n");
	return 0;
}
