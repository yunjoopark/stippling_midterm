#include "hedcut.h"

Hedcut::Hedcut()
{
	//control flags
	disk_size = 1;        //if uniform_disk_size is true, all disks have radius=disk_size,
	                      //othewise, the largest disks will have their radii=disk_size 

	uniform_disk_size = false; //true if all disks have the same size. disk_size is used when uniform_disk_size is true.
	black_disk = false;        //true if all disks are black ONLY

	//cvt control flags
	cvt_iteration_limit = 100; //max number of iterations when building cvf
	max_site_displacement = 1.01f; //max tolerable site displacement in each iteration. 

	debug = false;
}



bool Hedcut::build(cv::Mat & input_image, int n)
{
	cv::Mat input_GRAY;
	// YJ cvtColor(InputArrary src, OutputArray dst, int code, int dstCn = 0);
	cv::cvtColor(input_image, input_GRAY, CV_RGB2GRAY);

	//start = clock();
	//sample n points
	std::vector<cv::Point2d> pts;
	sample_initial_points(input_GRAY, n, pts);

	//initialize cvt
	CVT cvt;
	cvt.iteration_limit = this->cvt_iteration_limit;
	cvt.max_site_displacement = this->max_site_displacement;
	cvt.debug = this->debug;

	//compute weighted centroidal voronoi tessellation
	//finish = cvt.compute_weighted_cvt(input_image, pts);
	cvt.compute_weighted_cvt(input_GRAY, pts);

	//create disks
	create_disks(input_image, cvt);
	//finish = clock();
	//elapsed_time = (finish - start) / (double)CLOCKS_PER_SEC;
	return true;
}


void Hedcut::sample_initial_points(cv::Mat & img, int n, std::vector<cv::Point2d> & pts)
{
	//create n points that spread evenly that are in areas of black points...
	int count = 0;

	cv::RNG rng_uniform(time(NULL));
	cv::RNG rng_gaussian(time(NULL));
	cv::Mat visited(img.size(), CV_8U, cv::Scalar::all(0)); //all unvisited

	while (count < n)
	{
		//generate a random point
		int c = (int)floor(img.size().width*rng_uniform.uniform(0.f, 1.f));
		int r = (int)floor(img.size().height*rng_uniform.uniform(0.f, 1.f));

		//decide to keep basic on a probability (black has higher probability)
		float value = img.at<uchar>(r, c)*1.0/255; //black:0, white:1
		float gr = fabs(rng_gaussian.gaussian(0.8));
		if ( value < gr && visited.at<uchar>(r, c) ==0) //keep
		{
			count++;
			pts.push_back(cv::Point(r, c));
			visited.at<uchar>(r,c)=1;
		}
	}

	if (debug)
	{
		cv::Mat tmp = img.clone();
		for (auto& c : pts)
		{
			cv::circle(tmp, cv::Point(c.y, c.x), 2, CV_RGB(0, 0, 255), -1);
		}
		cv::imshow("samples", tmp);
		//cv::waitKey();
	}
}

void Hedcut::create_disks(cv::Mat & img, CVT & cvt)
{
	disks.clear();

	cv::Mat img_gray;
	cv::cvtColor(img, img_gray, CV_RGB2GRAY);

	if (debug)
	{
		cv::namedWindow("is gray", cv::WINDOW_AUTOSIZE);// Create a window for display.
		cv::imshow("is gray", img);                     // Show our image inside it.
		//cv::waitKey(0);                               // Wait for a keystroke in the window
	}

	//create disks from cvt
	for (auto& cell : cvt.getCells())
	{
		//compute avg intensity
		unsigned int total = 0;
		float total_r = 0;
		float total_g = 0;
		float total_b = 0;
		for (auto & pix : cell.coverage)
		{
			total += img_gray.at<uchar>(pix.x, pix.y);

			// YJ Vec3b intensity = img.at<Vec3b>(y, x);
			// uchar blue = intensity.val[0];
			// uchar green = intensity.val[1];
			// uchar red = intensity.val[2];
			total_b += img.at<cv::Vec3b>(pix.x, pix.y)[0];
			total_g += img.at<cv::Vec3b>(pix.x, pix.y)[1];
			total_r += img.at<cv::Vec3b>(pix.x, pix.y)[2];
		}
		float avg_v = floor(total * 1.0f/ cell.coverage.size());
		float r = floor(total_r * 1.0f / cell.coverage.size());
		float g = floor(total_g * 1.0f / cell.coverage.size());
		float b = floor(total_b * 1.0f / cell.coverage.size());

		//create a disk
		HedcutDisk disk;
		disk.center.x = cell.site.y; //x = col
		disk.center.y = cell.site.x; //y = row

		// YJ all(V) returns a scalar with all elements set to V0
		// YJ Scalar(a, b, c): a RGB color such as: Red = c, Green = b and Blue = a
		disk.color = (black_disk) ? cv::Scalar::all(0) : cv::Scalar(r, g, b);
		disk.radius = (uniform_disk_size) ? disk_size : (100 * disk_size / (avg_v + 100));

		//remember
		this->disks.push_back(disk);

	}//end for cell

	//done
}