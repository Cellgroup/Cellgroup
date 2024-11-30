/* H-Watershed segmentation
 * 
 * RESULT :- /Destination/EXP211_AO8_C13_aligned_SLICE_00X.png/*.zip => ROI.zip file
 * 
 * this macro only creates masks.  
*/
 
path = "path/A08/";
image = "EXP2111_A08_C13_aligned_SLICE_106.png";
destination = "path/A08/Destination/00H-Watershed/";
 
 /*  Define h and T values
 *   h = Increasing that value will merge regions, while decreasing it will split regions. 
 *   The reason is that logarithm scale provides a more intuitive interaction (many merging happen at low H but much fewer at large H.
 *   
 *   
 *   T = Intensity Threshold
 *   Increasing T decreases regions size while decreasing T till they eventually disappear.
 *   
 *   peak_flooding :- this parameter is an alternative way to explore regions thresholds. Rather than using a global threshold one can apply distinct thresholds to each segment. 
 *   Peak flooding allows to flood each peak till a certain percentage of its height. 
 *   For the region i the threshold T_i = T + \alpha/100 . (Imax_i-T), where T is the threshold parameter, \alpha is the peak flooding in percent and Imax_i is the maximum intensity in region i.
 *   
 *   allow_splitting = if checked, playing with threshold or peak flooding might create new region by splitting existing one. 
 *   If allow splitting is not checked the no new region is created, only the peak with the highest maxima is kept.
*/
// variables for performing stardist and H-Watershed, do not use it, if you are performing H-Watershed two times, instead use h1, T2, h2, T2

//////////////////////////////////////////////////////////////
/// and variables for Performing H-Watershed two times // uncomment these variables

h1 = 800;
T1 = 340;
peak_flooding1 = 90;
allow_splitting1 = false;

h2 = 50;
T2 = 350;
peak_flooding2 = 94;
allow_splitting2 = false;

//////////////////////////////////////////////////////////////////


// filter out some nuclei using these variables //
sizeMin = 25;
sizeMax = 800;
circularityMin = 0.00;
circularityMax = 1.00;

// Disable image display
setBatchMode(true);

// Check if the destination folder does not exist
if (!File.isDirectory(destination)) {
    File.makeDirectory(destination);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Checks whether the image has been segmented or not, if it is already segmented then it'll skip segmenting it //////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
if (!File.isDirectory(destination + image)) {
   
    run_Hwatershed(h1, T1, peak_flooding1 , allow_splitting1);		//h1, T1 , flooding, splitting
    //run_Hwatershed(h2, T2, peak_flooding2 , allow_splitting2);		//h1, T1 , flooding, split
	//combine_Hwatershed_result(h1, T1, h2, T2); //parameters are h1, T1, h2, T2
	
} else {
    print("Segmentation has already been performed on this image. Name conflict: a folder with the same name already exists in the destination folder.");
}



function run_Hwatershed(h, T, peak_flooding, allow_splitting) {
	
	// function to perform H-Watershed only. //
	
	open(path + image);
	//// Perform H-Watershed Segmentation ////
    selectWindow(image);
    print("Running H-Watershed...");
    run("H_Watershed", "impin=[" + getTitle() + "] hmin=" + h + " thresh=" + T + " peakflooding=" + peak_flooding + " outputmask=true allowsplitting=" + allow_splitting);

    // Iterate through the open images
    for (i = 1; i <= nImages; i++) {
        selectImage(i);
        title = getTitle();

        // Check if the title matches the desired pattern
        if (endsWith(title, "- watershed") && indexOf(title, "n=") >= 0) {
            // Select the desired window
            selectWindow(title);
            break;
        }
    }

    setOption("BlackBackground", true);
    run("Convert to Mask");
    run("Watershed");
    run("Fill Holes");
    run("Open");
    // Create the new folder
    File.makeDirectory(destination + image);
    saveAs("Tiff", destination + image + "/" + image + " - watershed (h=" + h + ", T=" + T + ", %=100).tif");
    print("Completed, Masks Saved to destination Folder");
    run("Analyze Particles...", "size=" + sizeMin + "-" + sizeMax + " circularity=" + circularityMin + "-" + circularityMax + " add");
    roiManager("Show All without labels");
    roiManager("Save", destination + image + "/" + image + "A" + ".zip");
    
    // Close every open window to free memory
    for (i = 1; i <= nImages; i++) {
        selectImage(i);
        close();
    }
}

function combine_Hwatershed_result(h1, T1, h2, T2) {
	// Performs BINARY XOR with masks, and then performs BINARY OR. 
	// Then analyze particles, we can adjust the size and circularity to filter out unwanted masks.
	print("combining masks");
	open(destination + "" + image + "/" + image + " - watershed (h=" + h1 + ", T=" + T1 + ", %=100).tif");
    selectWindow(image + " - watershed (h=" + h1 + ", T=" + T1 + ", %=100).tif");
    open(destination + "" + image + "/" + image + " - watershed (h=" + h2 + ", T=" + T2 + ", %=100).tif");
    selectWindow(image + " - watershed (h=" + h2 + ", T=" + T2 + ", %=100).tif");
    
    /////////    Image calculation XOR Watershed XOR with Stardist Mask ////
    imageCalculator("XOR create 32-bit", image + " - watershed (h=" + h1 + ", T=" + T1 + ", %=100).tif", image + " - watershed (h=" + h2 + ", T=" + T2 + ", %=100).tif");
    selectWindow("Result of " + image + " - watershed (h=" + h1 + ", T=" + T1 + ", %=100).tif");
    saveAs("Tiff", destination + image + "/" + image + "XOR_Mask.tif");
    close();
    open(destination + "" + image + "/" + image + "XOR_Mask.tif");
    selectWindow(image + "XOR_Mask.tif");
    setOption("BlackBackground", true);
    run("Convert to Mask");
    run("Watershed");
    run("Open");
    
    ///   Then Perform OR ////
    imageCalculator("OR create 32-bit", image + " - watershed (h=" + h1 + ", T=" + T1 + ", %=100).tif", image + "XOR_Mask.tif");
    selectWindow("Result of " + image + " - watershed (h=" + h1 + ", T=" + T1 + ", %=100).tif");
    run("Convert to Mask");
    run("Watershed");
    run("Open");
    run("Analyze Particles...", "size=" + sizeMin + "-" + sizeMax + " circularity=" + circularityMin + "-" + circularityMax + " add");
    roiManager("Show All without labels");
    roiManager("Save", destination + image + "/" + image + ".zip");
    
    // Close every open window to free Memory 
    for (i = 1; i <= nImages; i++) {
        selectImage(i);
        close();
    }
    
    print("completed, ROI.zip file saved to destination folder");
}


