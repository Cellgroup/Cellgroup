/* H-Watershed and stardist segmentation
 *  RESULT:- /Destination/EXP211_AO8_C13_aligned_SLICE_00X.png/*.zip => ROI.zip file

*/
path = "/Users/guidoputignano/Desktop/Stem cell Folder/A08/";
image = "EXP2111_A08_C13_aligned_SLICE_045.png";
destination = "/Users/guidoputignano/Desktop/Stem cell Folder/A08/Destination/";

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
h = 700.0;
T = 450.0;
peak_flooding = 98;
allow_splitting = true;

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
    runHWatershed_and_stardist(h, T, peak_flooding, allow_splitting);
    produceROI();
} else {
    print("Segmentation has already been performed on this image. Name conflict: a folder with the same name already exists in the destination folder.");
}

// define segmentation function
function runHWatershed_and_stardist(h, T, peak_flooding, allow_splitting){
	
	// function to perform H-Watershed and Stardist segmentation
	// In case We want to perform H-Watershed twice with different value we can adjust the parameter above, comment out the section of stardist and run this script twice with different values.//
	
	print("running segmentation on image " + image);
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
    run("Open");
    run("Watershed");
    // Create the new folder
    File.makeDirectory(destination + image);
    saveAs("Tiff", destination + image + "/" + image + " - watershed (h=" + h + ", T=" + T + ", %=100).tif");
    print("Completed, Masks Saved to destination Folder");

    /////////////////////////////////////////////////////////////////////////////////////////
    //// perform stardist segmentation //////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    selectWindow(image);
    //setOption("ScaleConversions", true);
    //run("Despeckle");
    //run("Enhance Contrast...", "saturated=0.35 normalize equalize");
    print("Starting Stardist Segmentation...");
    run("Command From Macro", "command=[de.csbdresden.stardist.StarDist2D], args=['input':'" + image + "', 'modelChoice':'Versatile (fluorescent nuclei)', 'normalizeInput':'true', 'percentileBottom':'1.0', 'percentileTop':'99.8', 'probThresh':'0.6000000000000001', 'nmsThresh':'0.8000000000000003', 'outputType':'Label Image', 'nTiles':'5', 'excludeBoundary':'2', 'roiPosition':'Automatic', 'verbose':'false', 'showCsbdeepProgress':'false', 'showProbAndDist':'false'], process=[false]");
    selectWindow("Label Image");
    setAutoThreshold("Otsu dark");
    //run("Threshold...");
    setThreshold(10, 65535, "raw");
    setOption("BlackBackground", true);
    run("Convert to Mask");
    run("Open");
    run("Watershed");
    //selectWindow("Label Image");
    saveAs("Tiff", destination + image + "/" + image + "_stardist_mask.tif");
    print("Completed, Masks Saved todestination Folder");

    // Close every open window to free memory
    for (i = 1; i <= nImages; i++) {
        selectImage(i);
        close();
    }
    print("completed segmentation");
}

// define the function that produces ROI.zip file
function produceROI() {
	
	// Performs BINARY XOR with masks, and then performs BINARY OR. 
	// Then analyze particles, we can adjust the size and circularity to filter out unwanted masks.
	
	// open two masks
	print("combining masks");
    open(destination + "" + image + "/" + image + " - watershed (h=" + h + ", T=" + T + ", %=100).tif");
    selectWindow(image + " - watershed (h=" + h + ", T=" + T + ", %=100).tif");
    open(destination + image + "/" + image + "_stardist_mask.tif");
    selectWindow(image + "_stardist_mask.tif");
    
    // perform XOR //
    imageCalculator("XOR create 32-bit", image + " - watershed (h=" + h + ", T=" + T + ", %=100).tif", image + "_stardist_mask.tif");
    selectWindow("Result of " + image + " - watershed (h=" + h + ", T=" + T + ", %=100).tif");
    saveAs("Tiff", destination + image + "/" + image + "XOR_Mask.tif");
    close();
    open(destination + "" + image + "/" + image + "XOR_Mask.tif");
    selectWindow(image + "XOR_Mask.tif");
    setOption("BlackBackground", true);
    run("Convert to Mask");
    run("Watershed");
    run("Open");
    run("Fill Holes");
    
    //Perform OR //
    imageCalculator("OR create 32-bit", image + " - watershed (h=" + h + ", T=" + T + ", %=100).tif", image + "XOR_Mask.tif");
    selectWindow("Result of " + image + " - watershed (h=" + h + ", T=" + T + ", %=100).tif");
    run("Convert to Mask");
    run("Watershed");
    run("Open");
    
    // analyze particles and generate ROI //
    run("Analyze Particles...", "size=" + sizeMin + "-" + sizeMax + " circularity="+ circularityMin +"-"+ circularityMax + " add");
    roiManager("Show All without labels");
    
    //save the results
    roiManager("Save", destination + image + "/" + image + ".zip");
    
    // Close every open window to free memory
    for (i = 1; i <= nImages; i++) {
        selectImage(i);
        close();
    }
    print("completed "+ image + ".zip file saved on destination folder");
}
