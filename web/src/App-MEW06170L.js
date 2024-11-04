// import logo from './logo.svg';
import './css/App.css';
import './css/portfolio.css';
import './css/bg.css';
import Portfolio from './components/Portfolio';
import Canvas from './components/Canvas';
import bgShapes1 from './static/Back Shapes_bg.svg';
import bgShapes2 from './static/Back Shapes_bg_2.svg';
import mvl from './static/madewithlove.svg';
import logo from './static/logo.svg';
import right from './static/rightarrow.svg';
// import done from './static/done.svg';
// import pending from './static/pending.svg';
// import main from './static/main img.svg';
import { MathJaxContext, MathJax } from 'better-react-mathjax';
import SyntaxHighlighter from 'react-syntax-highlighter';
import { tomorrowNightBright } from 'react-syntax-highlighter/dist/esm/styles/hljs';
import { useState, useEffect, useRef } from 'react';
import {
  rgbToHslHsvHex,
  hexToRgb,
  hsvRgbObjToArr,
  getColourPoints,
} from './js/colorFunctions';

import CODA from './images/dataset.jpeg';
import structure from './images/gaussian.jpeg';
// import goldberg from './images/octave.jpeg';
import DoG from './images/DoG.png';
import Gaussian from './images/Gaussian.png';
import Resample from './images/Resample.png';
import main_img from './images/Head.png';
import LP from './images/LP.png';

function useInterval(callback, delay) {
  const savedCallback = useRef();

  // Remember the latest callback.
  useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);

  // Set up the interval.
  useEffect(() => {
    function tick() {
      savedCallback.current();
    }
    if (delay !== null) {
      let id = setInterval(tick, delay);
      return () => clearInterval(id);
    }
  }, [delay]);
}

function App() {
  var rawCP = {
    colours: [
      '0f1e17',
      '011d26',
      '080b0f',
      '000000',
      '000000',
      '003d1e',
      '0f1e17',
    ],
    // colours: ['000000','FF0000','00FF00','0000FF','FFFF00','00FFFF','FF00FF'],
    radii: [1.3, 1.9, 1.53, 1.01, 1.81, 0.6, 1.11],
    positions: [
      [120.8854, 78.9364],
      [382.0792, 303.7385],
      [108.4034, 492.5733],
      [483.9389, 561.1634],
      [816.4031, 521.2621],
      [832.2423, 592.2427],
      [810.4892, 299.7712],
    ],
    viewport: [841.89, 595.28],
  };
  const [colourPoints, setColourPoints] = useState(getColourPoints(rawCP));
  const [oCP, setOCP] = useState(JSON.parse(JSON.stringify(colourPoints)));
  const [startPerturbation, setStartPerturbation] = useState(false);
  const printColour = (p) => {
    console.log([p.x, p.y]);
  };
  const perturbPos = (first = false) => {
    let cP = JSON.parse(JSON.stringify(colourPoints));
    // console.log(
    //     Math.max(Math.min(oCP[0].x + (Math.random() * 2 - 1) * 1, 1))
    // );
    if (first) {
      for (let i in oCP) {
        oCP[i].x = Math.max(
          Math.min(oCP[i].x + ((Math.random() * 2 - 1) / 2) * 0.5, 1),
          0
        );
        oCP[i].y = Math.max(
          Math.min(oCP[i].y + ((Math.random() * 2 - 1) / 2) * 0.5, 1),
          0
        );
        setOCP([...oCP]);
        // // console.log([cP[i].x, cP[i].y]);
        // cP[i].x = cP[i].x + (Math.random() * 2 - 1) * 0.1;
        // cP[i].y = cP[i].y + (Math.random() * 2 - 1) * 0.1;
        // console.log([cP[i].x, cP[i].y]);
      }
    } else if (startPerturbation) {
      for (let i in oCP) {
        // console.log([cP[i].x, cP[i].y]);
        // if (
        //     oCP[i].x - cP[i].x < 0.0001 ||
        //     oCP[i].y - cP[i].y < 0.0001
        // ) {
        oCP[i].x =
          (((oCP[i].x + ((Math.random() * 2 - 1) / 2) * 0.05) % 1) + 1) % 1;
        oCP[i].y =
          (((oCP[i].y + ((Math.random() * 2 - 1) / 2) * 0.05) % 1) + 1) % 1;
        setOCP([...oCP]);
        // }
        cP[i].x = Math.max(
          Math.min(cP[i].x + 0.005 * Math.random() * (oCP[i].x - cP[i].x), 1),
          0
        );
        cP[i].y = Math.max(
          Math.min(cP[i].y + 0.005 * Math.random() * (oCP[i].y - cP[i].y), 1),
          0
        );
        // console.log(0.05 * Math.random() * (oCP[i].y - cP[i].y));
        // console.log([cP[i].x, cP[i].y]);
      }
    }
    setColourPoints(cP);
  };

  useInterval(perturbPos, 1);
  // console.log(SyntaxHighlighter.supportedLanguages)
  useEffect(() => {
    perturbPos(true);
    setStartPerturbation(true);
  }, []);

  return (
    <MathJaxContext>
      <div className="App">
        <div class="backgrounds">
          <Canvas id={'gradientPalette'} canvasPoints={colourPoints} />
          <div className="bg_c_c">
            <div id="bg_Container_1" className="bg_Container">
              <object
                data={bgShapes1}
                id="bgShapes_1"
                type="image/svg+xml"
                alt="bg image"
              />
            </div>
          </div>
          <div className="bg_c_c">
            0
            <div id="bg_Container_2" className="bg_Container">
              <object
                data={bgShapes2}
                id="bgShapes_2"
                type="image/svg+xml"
                alt="another bg image"
              />
            </div>
          </div>
          <object
            data={logo}
            id="logo"
            type="image/svg+xml"
            alt="spiral logo"
          />
          <p id="myname">
            Let's OpenC<span className="light small">UDA</span>V
          </p>
          <object
            data={mvl}
            id="mvl"
            type="image/svg+xml"
            alt="made with love"
          />
        </div>
        <div className="main-box">
          <div className="main-content">
            <div id="pf_con_con">
              <div id="pfs_container" className="medium">
                <p className="pfTitle">Idea</p>
                <div className="main-img-box">
                  {/* <object
                    data={main}
                    className="main"
                    type="image/svg+xml"
                    alt="spiral logo"
                  /> */}
                  <img src={main_img} alt="CUDA+SIFT?" />
                </div>

                <div className="flex ttb full-width center">
                  <h2 className="pfTitle half-bottom-margin">
                    A GPU-Accelerated Fast SIFT
                  </h2>

                  <p className="pfContent width-control center-align no-top-margin">
                    {/* Paragraph 1 */}A Scale-Invariant Feature Transform
                    (SIFT) for CUDA enabled devices, with memory optimization
                    and scope for large parallelization.
                  </p>
                </div>
              </div>
            </div>

            <div id="pf_con_con">
              <div id="pfs_container" className="medium">
                <div id="introduction_section" className="medium">
                  <p className="pfTitle">Introduction</p>
                  <div className="row">
                    <div>
                      <p>
                        <span className="head"></span>
                      </p>
                      <p>{/* Your introduction content goes here */}</p>
                    </div>
                  </div>
                </div>
                <p className="pfContent width-control">
                  {/* Paragraph 1 */}
                  Our project focuses on accelerating the Scale-Invariant
                  Feature Transform (SIFT) feature extraction process, a
                  widely-used methodology for identifying interest points in
                  images and videos. We aim to reduce the time required for this
                  task by parallelizing various stages of the SIFT algorithm
                  using CUDA-enabled GPUs. This approach substantially boosts
                  the efficiency of video analysis, making it faster and more
                  scalable compared to traditional methods.
                </p>
                <p className="pfContent width-control">
                  The motivation for our project arises from the growing demand
                  for efficient video analysis across industries such as
                  surveillance, entertainment, and healthcare. Our
                  GPU-accelerated solution addresses this need by delivering
                  significant efficiency gains, leading to cost savings and
                  improved productivity for organizations that rely on video
                  analysis for decision-making, monitoring, or research
                  purposes.
                </p>
                {/* Paragraph 3 */}
                <p className="pfContent width-control">
                  For our system, the expected input is a video file, and the
                  output consists of the coordinates or descriptors of the
                  interest points detected in each frame. Specifically, our
                  system will generate a list of interest points along with
                  their corresponding features for each frame. This output
                  facilitates further analysis, object recognition, or tracking
                  tasks, enabling researchers, analysts, and professionals to
                  extract valuable insights from videos more efficiently. By
                  improving the speed and accuracy of video analysis, we aim to
                  enhance the overall efficiency and productivity of video
                  analysis workflows.
                </p>
                <p className="pfContent width-control">
                  <a
                    href="https://github.gatech.edu/kshenoy8/stereosquad/"
                    target="_blank"
                  >
                    See the GitHub Repository here!
                  </a>
                </p>
              </div>
            </div>
            <div id="pf_con_con">
              <div id="pfs_container" className="medium">
                <p className="pfTitle">Approach</p>
                <div className="row row-left">
                  <div className="img img-left">
                    <img src={CODA} alt="hello" />
                  </div>
                  <div>
                    <p>
                      <span className="head">
                        <span className="lime">1 </span> Capturing Videos for
                        Dataset Creation
                      </span>
                    </p>
                    <p>
                      {' '}
                      {/* Epipolar geometry is fundamental to our project’s goal of
                      generating stereoscopic views from single-camera videos.
                      It involves understanding the geometric relationship
                      between multiple views of the same scene, which is crucial
                      for 3D reconstruction. Using epipolar geometry, we can
                      determine corresponding points between frames, aiding in
                      the estimation of depth and facilitating the creation of a
                      stereoscopic effect. */}
                      We curated our own dataset for the project, focusing on
                      the Coda building. To maintain clarity and sharpness
                      across the frames of the video, we used a gimbal during
                      recording. This ensured minimal blurring and stable
                      footage throughout, enhancing the quality of the video
                      dataset. For accurate camera calibration, we use various
                      checkerboard patterns. These patterns serve as a
                      calibration tool due to their well-defined geometry and
                      known dimensions. The known properties of the checkerboard
                      patterns provides a ground truth reference helping in the
                      precise estimation of camera parameters. To diversify our
                      calibration dataset and avoid symmetrical patterns that
                      could potentially skew the calibration results, we
                      designed custom 3x3 grids.
                    </p>
                  </div>
                </div>
                <div className="row row-right">
                  <div>
                    <p>
                      <span className="head">
                        <span className="lime">2 </span> Gaussian Filtering and
                        Separability
                      </span>
                    </p>
                    <p>
                      In our project, we utilized a Gaussian filter for image
                      blurring due to its separable nature, which offers
                      computational advantages. Traditional 2D convolution
                      filters require nxm multiplications for each pixel, where
                      n and m are the dimensions of the kernel. In contrast,
                      separable filters can be split into two one-dimensional
                      filters applied sequentially along rows and columns. By
                      taking advantage of this property, we divided the blurring
                      process into horizontal and vertical stages. This method
                      reduces the computational load to just n + m
                      multiplications per pixel, making the Gaussian filter an
                      efficient choice for image blurring while maintaining
                      quality.
                      {/* Smith et al. utilize HDR cameras mounted on a mobile robot
                      for stereo vision-based 3D reconstruction{' '}
                      <span className="lime">[1]</span>. Their method captures
                      textures and spatial features as 2D images and employs an
                      algorithm for depth map visualization.
                      <br />
                      <br />
                      OpenMVG is a library dedicated to Multiple-View Geometry
                      and Structure-from-Motion tasks{' '}
                      <span className="lime">[3]</span>. It facilitates
                      identifying corresponding points between images and
                      manipulating 3D geometry for various computer vision
                      applications. */}
                    </p>
                    <SyntaxHighlighter
                      language="julia"
                      style={tomorrowNightBright}
                      showLineNumbers={true}
                      wrapLines={true}
                      className="code"
                    >
                      {`function col_kernel_strips(inp, conv, buffer, width::Int32, height::Int16, apron::Int8)
    # block number, column major, 0-indexed
    blockNum::UInt32 = blockIdx().x - 1 + 
                        (blockIdx().y - 1) * gridDim().x 
    threadNum::UInt16 = threadIdx().x - 1
    threads::Int16 = blockDim().x

    # there could be more blocks than needed
    thisX::Int32 = blockNum ÷ cld((height - 2 * apron), 
                                (threads - 2 * apron)) + 1 # 1-indexed
    thisY::Int16 = blockNum % cld((height - 2 * apron), 
                                (threads - 2 * apron)) 
                            * (threads - 2 * apron) + threadNum + 1 # 1-indexed
    thisPX::Int32 = 0

    data = CuDynamicSharedArray(Float32, threads)

    # fill the shared memory
    if thisY <= height && thisX <= width
        thisPX = thisY + (thisX - 1) * height
        data[threadNum+1] = inp[thisPX]
    end
    sync_threads()

    # convolution
    if apron < thisY <= height - apron && thisX <= width && apron <= threadNum < threads - apron
        sum::Float32 = 0.0
        for i in -apron:apron
            sum += data[threadNum+1+i] * conv[apron+1+i]
        end
        buffer[thisY, thisX] = sum
    end
    return
end

# buffH is the height of the buffer including the black apron at the bottom
# inpH is the height of the image excluding the aprons, after the column kernel
function row_kernel(inp, conv, out, inpH::Int16, buffH::Int16, width::Int32, imgWidth::Int16, apron::Int8)
    # block number, column major, 0-indexed
    blockNum::UInt32 = blockIdx().x - 1 
                        + (blockIdx().y - 1) 
                        * gridDim().x 
    threadNum::UInt16 = threadIdx().x - 1 
                        + (threadIdx().y - 1) 
                        * blockDim().x
    threads::Int16 = blockDim().x * blockDim().y

    if threads <= width

        blocksInACol::Int8 = cld(inpH, blockDim().x)
        blocksInARow::Int16 = cld(imgWidth - 2 * apron, blockDim().y - 2 * apron)
        blocksInAnImage::Int16 = blocksInACol * blocksInARow

        thisX::Int32 = (blockNum ÷ blocksInAnImage) * imgWidth + ((blockNum % blocksInAnImage) % blocksInARow) * (blockDim().y - 2 * apron) + threadIdx().y # 1-indexed
        thisY::Int16 = ((blockNum % blocksInAnImage) ÷ blocksInARow) * blockDim().x + threadIdx().x + apron # 1-indexed

        data = CuDynamicSharedArray(Float32, (blockDim().x, blockDim().y))

        # fill the shared memory
        thisPX::Int32 = thisY + (thisX - 1) * buffH
        if thisX <= width && thisY <= inpH + apron
            data[threadNum+1] = inp[thisPX]
        end
        sync_threads()

        # convolution
        thisIsAComputationThread::Bool = 
                  thisY <= inpH + apron 
                    && apron < thisX <= width - apron 
                    && apron < threadIdx().y <= blockDim().y - apron

        if (blockNum % blocksInAnImage) % blocksInARow == blocksInARow - 1
            thisIsAComputationThread = 
                    thisIsAComputationThread 
                  && (thisX - (blockNum ÷ blocksInAnImage) * imgWidth <= imgWidth - 2 * apron)
        end
        if thisIsAComputationThread
            sum::Float32 = 0.0
            for i in -apron:apron
                sum += data[threadNum+1+i*blockDim().x] * conv[apron+1+i]
            end
            # out[thisY, thisX-apron-fld(blockNum, blocksInAnImage)*2*apron] = sum
            out[thisY, thisX] = sum
        end
    end
    return
end`}
                    </SyntaxHighlighter>
                  </div>
                  <div className="img img-right">
                    <img src={Gaussian} alt="hello" />
                  </div>
                </div>
                <div className="row row-left">
                  <div className="img img-left">
                    <img src={DoG} alt="hello" />
                  </div>
                  <div>
                    <p>
                      <span className="head">
                        <span className="lime">3 </span>
                        Difference of Gaussian (DoG)
                      </span>
                    </p>
                    <p>
                      The Difference of Gaussian (DoG) method is employed to
                      effectively identify stable keypoint locations within the
                      scale space of an image. This is achieved by convolving
                      the image with the difference-of-Gaussian function,{' '}
                      <MathJax className="math" inline={true}>
                        {'\\(\\text{D}(x, y, σ)\\)'}
                      </MathJax>{' '}
                      which is derived from the difference between two
                      neighboring scales separated by a constant multiplicative
                      factor{' '}
                      <MathJax className="math" inline={true}>
                        {'\\(k\\)'}
                      </MathJax>
                      . One of the primary advantages of using this function is
                      its computational efficiency. As the smoothed images{' '}
                      <MathJax className="math" inline={true}>
                        {'\\(L\\)'}
                      </MathJax>{' '}
                      are already computed for scale space feature description,{' '}
                      <MathJax className="math" inline={true}>
                        {'\\(D\\)'}
                      </MathJax>{' '}
                      can be easily obtained through straightforward image
                      subtraction.
                      <br />
                      <br />
                      Moreover, the DoG function closely approximates the
                      scale-normalized Laplacian of Gaussian. In practice, the
                      initial image undergoes incremental convolution with
                      Gaussians to generate images at different scales, spaced
                      by a constant factor k n scale space. For efficient octave
                      processing, each octave is divided into s intervals,
                      setting,{' '}
                      <MathJax className="math">
                        {'$$k = 2^{1/s}.$$'}
                      </MathJax>{' '}
                      To cover a complete octave during extrema detection, we
                      produce{' '}
                      <MathJax className="math" inline={true}>
                        {'\\((s+3)\\)'}
                      </MathJax>{' '}
                      images in the blurred image stack for each octave. The DoG
                      images, resulting from subtracting adjacent scales, are
                      shown on the left. After processing a full octave, the
                      Gaussian image with double the initial{' '}
                      <MathJax className="math" inline={true}>
                        {'\\(\\sigma\\)'}
                      </MathJax>{' '}
                      value is resampled by selecting every second pixel in each
                      row and column. This resampling maintains accuracy
                      relative to{' '}
                      <MathJax className="math" inline={true}>
                        {'\\(\\sigma\\)'}
                      </MathJax>{' '}
                      while significantly reducing computational overhead.
                    </p>
                    <SyntaxHighlighter
                      language="julia"
                      style={tomorrowNightBright}
                      showLineNumbers={true}
                      wrapLines={true}
                      className="code"
                    >
                      {`for i in 1:(layers-1)
    out_gpus[j][i] = out_gpus[j][i+1] .- out_gpus[j][i]
    out_gpus[j][i] = out_gpus[j][i] .* (out_gpus[j][i] .> 0.0)
end`}
                    </SyntaxHighlighter>
                  </div>
                </div>

                <div className="row row-right">
                  <div>
                    <p>
                      <span className="head">
                        <span className="lime">4 </span> Resampling for Octaves
                      </span>
                    </p>
                    <p>
                      We resample between each octave to create a series of
                      images with different scales. This is done to detect
                      features at different levels of detail, making the
                      algorithm invariant to scale changes in the image.
                    </p>
                    <p>
                      The image with{' '}
                      <MathJax className="math" inline={true}>
                        {'\\(2\\times\\)'}
                      </MathJax>{' '}
                      the base{' '}
                      <MathJax className="math" inline={true}>
                        {'\\(\\sigma\\)'}
                      </MathJax>{' '}
                      is used as the source to resample because it ensures that
                      the keypoint detection is efficient and robust. This
                      choice allows SIFT to detect a wide range of features at
                      different scales while minimizing computational
                      complexity. The{' '}
                      <MathJax className="math" inline={true}>
                        {'\\(2\\times\\)'}
                      </MathJax>
                      <MathJax className="math" inline={true}>
                        {'\\(\\sigma\\)'}
                      </MathJax>{' '}
                      image helps capture both finer and coarser details,
                      ensuring the algorithm is robust across different scales.{' '}
                      <br />
                      <br />
                    </p>
                    <SyntaxHighlighter
                      language="julia"
                      style={tomorrowNightBright}
                      showLineNumbers={true}
                      wrapLines={true}
                      className="code"
                    >
                      {`function resample_kernel(inp, out)
    blockNum::UInt32 = blockIdx().x - 1 + (blockIdx().y - 1) * gridDim().x 
    # block number, column major, 0-indexed
    threadNum::UInt16 = threadIdx().x - 1
    threads::Int16 = blockDim().x

    data = CuDynamicSharedArray(Float32, threads)

    h, w = size(inp)
    outPX::Int32 = blockNum * threads + threadNum + 1
    outX::Int32 = (outPX - 1) ÷ (h ÷ 2) # 0-indexed
    outY::Int16 = (outPX - 1) % (h ÷ 2) # 0-indexed

    thisX::Int32 = 2 * outX # 0-indexed
    thisY::Int16 = 2 * outY # 0-indexed
    thisPX::Int32 = thisY + thisX * h + 1

    # fill the shared memory
    if thisPX <= h * w
        data[threadNum+1] = inp[thisPX]
    end
    sync_threads()

    if outPX <= (h * w) ÷ 4
        out[outPX] = data[threadNum+1]
    end
    return
end`}
                    </SyntaxHighlighter>
                    {/* Smith et al. utilize HDR cameras mounted on a mobile robot
                      for stereo vision-based 3D reconstruction{' '}
                      <span className="lime">[1]</span>. Their method captures
                      textures and spatial features as 2D images and employs an
                      algorithm for depth map visualization.
                      <br />
                      <br />
                      OpenMVG is a library dedicated to Multiple-View Geometry
                      and Structure-from-Motion tasks{' '}
                      <span className="lime">[3]</span>. It facilitates
                      identifying corresponding points between images and
                      manipulating 3D geometry for various computer vision
                      applications. */}
                  </div>
                  <div className="img img-right">
                    <img src={Resample} alt="hello" />
                  </div>
                </div>
                <div className="row row-left">
                  {' '}
                  <div className="img img-left">
                    <img src={LP} alt="hello" />
                  </div>
                  <div>
                    <p>
                      <span className="head">
                        <span className="lime">5 </span> Large Parallelism
                      </span>
                    </p>
                    <p>
                      In order to make our code more efficient, we utilized
                      large parallelism to process multiple images/frames
                      simultaneously. This approach allows us to take advantage
                      of the high memory bandwidth of modern GPUs, significantly
                      reducing the time required for image processing.
                    </p>
                    <p>
                      By processing 128 images at once, we were able to achieve
                      optimal performance and enhance the speed of our SIFT
                      algorithm. This also makes the code more scalable to
                      handle hardware with higher memory bandwidths and
                      processing capabilities in the future. Processing multiple
                      images in parallel is a nice way to process a video
                      because videos are essentially a series of images. This
                      parallel processing capability is crucial for accelerating
                      the feature extraction process and improving the
                      efficiency of video analysis tasks.
                    </p>
                    <SyntaxHighlighter
                      language="julia"
                      style={tomorrowNightBright}
                      showLineNumbers={true}
                      wrapLines={true}
                      className="code"
                    >
                      {`nImages = 128
  img = []
  imgWidth = 0
  time_taken = 0
  # load the images
  for i in 1:nImages
      img_temp = Float32.(Gray.(FileIO.load("assets/frame_$i.png")))
      if i == 1
          img = img_temp
          imgWidth = size(img, 2)
      else
          img = cat(img, img_temp, dims=2)
      end
  end`}
                    </SyntaxHighlighter>
                    {/* Smith et al. utilize HDR cameras mounted on a mobile robot
                      for stereo vision-based 3D reconstruction{' '}
                      <span className="lime">[1]</span>. Their method captures
                      textures and spatial features as 2D images and employs an
                      algorithm for depth map visualization.
                      <br />
                      <br />
                      OpenMVG is a library dedicated to Multiple-View Geometry
                      and Structure-from-Motion tasks{' '}
                      <span className="lime">[3]</span>. It facilitates
                      identifying corresponding points between images and
                      manipulating 3D geometry for various computer vision
                      applications. */}
                  </div>
                </div>
              </div>
            </div>
            <div id="pf_con_con">
              <div id="pfs_container" className="medium">
                <p className="pfTitle">Experiment Setup</p>
                <div className="row">
                  <div>
                    <p>
                      <span className="head">
                        <span className="lime">1 </span> CUDA and Occupancy API
                      </span>
                    </p>
                    <p>
                      We first attempted to run 1024 threads per block. However,
                      some kernels, such as the row convolution kernel, were not
                      able to run with this configuration. We then tried
                      lowering the threads per block, which worked for all
                      kernels. We also experimented with different block sizes
                      and found that 16x48 (768 threads per block) was the
                      optimal block size for row kernel. This is because the
                      number of times apron elements are loaded into shared
                      memory is minimized, and the number of threads that are
                      actually used is maximized. It also allows for a half-warp
                      memory coallescing, making it more efficient for older and
                      while still being efficient for newer GPUs.{' '}
                    </p>
                    <p>
                      Although the grid dimensions do not matter, we used a
                      nearly square grid for all kernel to maximize the
                      dimension of a grid since there is a hard limit on the
                      number of blocks in each direction.
                    </p>
                    <p>
                      We used the occupancy API to determine the optimal block
                      size for each kernel. The occupancy API provides
                      information about the maximum number of threads that can
                      be run on a multiprocessor, helping us optimize the
                      performance of our CUDA kernels. CUDA calls are shown
                      below.
                    </p>
                    <SyntaxHighlighter
                      language="julia"
                      style={tomorrowNightBright}
                      showLineNumbers={true}
                      wrapLines={true}
                      className="code medium long"
                    >
                      {`function doLayersConvolvesAndDoGAndOctave(img_gpu, out_gpus, buffer, conv_gpus, aprons, height, width, imgWidth, layers, octaves)
    time_taken = 0
    for j in 1:octaves
        for i in 1:layers
            threads_column = 1024 #32 * 32
            threads_row = (16, 768 ÷ 16)
            while threads_row[2] - 2 * aprons[i] <= 0 && threads_row[1] > 4
                threads_row = (threads_row[1] ÷ 2, threads_row[2] * 2)
            end

            if cld(height, prod(threads_column)) >= 1
                blocks_column = makeThisNearlySquare((cld(height - 2 * aprons[i], threads_column - 2 * aprons[i]), width))
                blocks_row = makeThisNearlySquare((cld(height - 2 * aprons[i], threads_row[1]) * cld(width - 2 * aprons[i], threads_row[2] - 2 * aprons[i]) + cld(height - 2 * aprons[i], threads_row[1]) / 2 * cld(imgWidth - 2 * aprons[i], threads_row[2] - 2 * aprons[i]), 1))
                shmem_column = threads_column * sizeof(Float32)
                shmem_row = threads_row[1] * threads_row[2] * sizeof(Float32)

                time_taken += CUDA.@elapsed buffer .= 0
                time_taken += CUDA.@elapsed @cuda threads = threads_column blocks = blocks_column shmem = shmem_column col_kernel_strips(img_gpu, conv_gpus[i], buffer, Int32(width), Int16(height), Int8(aprons[i]))
                time_taken += CUDA.@elapsed @cuda threads = threads_row blocks = blocks_row shmem = shmem_row row_kernel(buffer, conv_gpus[i], out_gpus[j][i], Int16(height - 2 * aprons[i]), Int16(height), Int32(width), Int16(imgWidth), Int8(aprons[i]))
            end
        end
        time_taken += CUDA.@elapsed buffer = CUDA.zeros(Float32, cld(height, 2), cld(width, 2))
        time_taken += CUDA.@elapsed img_gpu = CUDA.zeros(Float32, cld(height, 2), cld(width, 2))
        time_taken += CUDA.@elapsed @cuda threads = 1024 blocks = makeThisNearlySquare((cld(height * width ÷ 4, 1024), 1)) shmem = 1024 * sizeof(Float32) resample_kernel(out_gpus[j][3], img_gpu)
        for i in 1:(layers-1)
            time_taken += CUDA.@elapsed out_gpus[j][i] = out_gpus[j][i+1] .- out_gpus[j][i]
            time_taken += CUDA.@elapsed out_gpus[j][i] = out_gpus[j][i] .* (out_gpus[j][i] .> 0.0)
        end
        height = height ÷ 2
        width = width ÷ 2
    end
    return time_taken
end`}
                    </SyntaxHighlighter>
                    <p>Occupancy API is used like so.</p>
                    <SyntaxHighlighter
                      language="julia"
                      style={tomorrowNightBright}
                      showLineNumbers={true}
                      wrapLines={true}
                      className="code medium"
                    >
                      {`kernel = @cuda name = "col" launch = false col_kernel_strips(img_gpu, conv_gpus[1], buffer, Int32(width), Int16(height), Int8(aprons[i]))
println(launch_configuration(kernel.fun))
kernel = @cuda name = "row" launch = false row_kernel(buffer, conv_gpus[i], out_gpus[j][i], Int16(height - 2 * aprons[i]), Int16(height), Int32(width), Int16(imgWidth), Int8(aprons[i]))
println(launch_configuration(kernel.fun))`}
                    </SyntaxHighlighter>
                    <p>
                      And finally, the grids are made nearly square like so.
                    </p>
                    <SyntaxHighlighter
                      language="julia"
                      style={tomorrowNightBright}
                      showLineNumbers={true}
                      wrapLines={true}
                      className="code medium"
                    >
                      {`function makeThisNearlySquare(blocks)
    product = prod(blocks)
    X = floor(Int32, sqrt(product))
    Y = X
    while product % X != 0 && X / Y > 0.75
        X -= 1
    end

    if product % X == 0
        return Int32.((X, product ÷ X))
    else
        return Int32.((Y, cld(product, Y)))
    end
end`}
                    </SyntaxHighlighter>
                  </div>
                </div>
                <div className="row">
                  <div>
                    <p>
                      <span className="head">
                        <span className="lime">2 </span> Gaussian Kernel and
                        Cut-off
                      </span>
                    </p>
                    <p>
                      The Gaussian kernel is a fundamental component of the SIFT
                      algorithm, used for image blurring and feature detection.
                      The kernel is defined by the Gaussian function, which is a
                      bell-shaped curve that represents the distribution of
                      values across the image. The Gaussian kernel is applied to
                      the image to smooth out noise and reduce the impact of
                      high-frequency components. This process helps in
                      identifying stable keypoints and features in the image,
                      making it easier to detect and match interest points
                      across different scales.
                    </p>
                    <p>
                      However, the Gaussian kernel has a cut-off point beyond
                      which the values are negligible. This cut-off point is
                      determined by the standard deviation of the Gaussian
                      function, which controls the spread of the kernel. By
                      setting an appropriate cut-off value, we can limit the
                      size of the kernel and reduce the computational load. This
                      optimization ensures that the convolution operation is
                      efficient and fast, making the SIFT algorithm more
                      scalable and practical for real-world applications.
                    </p>
                    <p>
                      In our code, we implemented the Gaussian kernel such that
                      the outermost un-normalized value is 0.1725.
                      <MathJax className="math" inline={true}>
                        {'$$\\epsilon=0.1725\\le e^{-X^2/(2\\sigma^2)}$$'}
                      </MathJax>{' '}
                      <MathJax className="math" inline={true}>
                        {
                          '$$\\Rightarrow \\text{apron}=\\sigma \\cdot \\sqrt{2\\ \\text{ln} (1/\\epsilon)} $$'
                        }
                      </MathJax>
                      This ensures that the kernel is normalized and that the
                      values beyond a certain point are negligible.
                    </p>
                    <SyntaxHighlighter
                      language="julia"
                      style={tomorrowNightBright}
                      showLineNumbers={true}
                      wrapLines={true}
                      className="code medium"
                    >
                      {`schema = Dict(:name => "gaussian1D", :epsilon => 0.1725, :sigma => 1.6)
sigma = convert(Float64, schema[:sigma])
epsilon = haskey(schema, :epsilon) ? schema[:epsilon] : 0.0001
apron = ceil(Int, sigma * sqrt(-2 * log(epsilon)))`}
                    </SyntaxHighlighter>
                  </div>
                </div>
                <div className="row">
                  <div>
                    <p>
                      <span className="head">
                        {/* <span className="lime">1 </span>Structure from Motion */}
                      </span>
                    </p>
                    <p>
                      In our experiments, we generated a dataset comprising 1980
                      frames extracted from a 66-second video. To efficiently
                      apply convolution with Gaussian blurring in both vertical
                      and horizontal directions, we processed 128 images
                      simultaneously, arranged in a row. This approach was
                      designed to harness the high DRAM availability of modern
                      GPUs. Specifically, we utilized an NVIDIA H100 80GB HBM3
                      GPU from the Georgia Tech's PACE cluster. With CUDA
                      Version 12.2 and Julia 1.9.2, we leveraged the parallel
                      processing capabilities of our GPU to achieve optimal
                      performance.
                    </p>
                  </div>
                </div>
                <div className="row">
                  <div></div>
                </div>
                <div className="row">
                  <div>
                    <p></p>
                  </div>
                </div>
              </div>
            </div>

            <div id="pf_con_con">
              <div id="pfs_container" className="short">
                <p className="pfTitle">Results</p>
                <div className="row">
                  <div>
                    <p>
                      <span className="head">
                        {/* <span className="lime">1 </span>Structure from Motion */}
                      </span>
                    </p>
                    <p>
                      In our experiments, we generated a dataset comprising 1980
                      frames extracted from a 66-second video. To efficiently
                      apply convolution with Gaussian blurring in both vertical
                      and horizontal directions, we processed 128 images
                      simultaneously, arranged in a row. This approach was
                      designed to harness the high DRAM availability of modern
                      GPUs. Specifically, we utilized an NVIDIA H100 80GB HBM3
                      GPU from the Georgia Tech's PACE cluster. With CUDA
                      Version 12.2 and Julia 1.9.2, we leveraged the parallel
                      processing capabilities of our GPU to achieve optimal
                      performance.
                    </p>
                  </div>
                </div>

                <div className="row">
                  <div>
                    <p>
                      <span className="head">
                        <span className="lime">3 </span> Interest Points
                      </span>
                    </p>
                    <p>
                      Although our approach primarily relies on interest points
                      being necessarily vertices, we are also considering the
                      possibility of using lines and their angles as another
                      "feature"-set. This will probably allow us to generate
                      surfaces using triangular meshes and will be a more robust
                      approach.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div id="pf_con_con">
              <div id="pfs_container" className="short">
                <p className="pfTitle">Challenges</p>
                <div className="row">
                  <div>
                    <p>
                      <span className="head">
                        {/* <span className="lime">1 </span>Structure from Motion */}
                      </span>
                    </p>
                    <p>
                      Throughout our project, we encountered several challenges
                      that influenced our approach and focus. Initially, one of
                      the major challenges was obtaining access to the newly
                      installed H100 GPUs on the pace cluster. Setting up the
                      environment with compatible versions of Julia and CUDA was
                      also challenging due to memory limitations. These initial
                      hurdles delayed our progress and required troubleshooting
                      to ensure a stable development environment.
                    </p>
                    <p>
                      Another challenge was aligning our project goals with the
                      reality of processing large videos. Our initial proposal
                      aimed to generate a stereo video from a single camera's
                      output. However, optimizing the SIFT algorithm for videos
                      with over 1000 frames proved to be more time-consuming
                      than anticipated. As a result, we shifted our focus to
                      optimizing the SIFT algorithm for large video datasets,
                      achieving excellent results in terms of speed and
                      efficiency.
                    </p>
                  </div>
                </div>
                <p className="pfTitle">Future Work</p>
                <div className="row">
                  <div>
                    <p>
                      <span className="head">
                        {/* <span className="lime">1 </span>Structure from Motion */}
                      </span>
                    </p>
                    <p>
                      While we couldn't complete the entire project as initially
                      planned, there are several areas we identified for future
                      work:
                    </p>
                    <p>
                      <span className="head">
                        <span className="lime">1 </span> Feature Mapping
                      </span>
                    </p>
                    <p>
                      Implementing KD-tree algorithms for efficient feature
                      mapping across different frames of the video, focusing on
                      handling high-dimensional data efficiently.
                    </p>
                  </div>
                </div>
                <div className="row">
                  <div>
                    <p>
                      <span className="head">
                        <span className="lime">2 </span>Fundamental Matrix
                        Computation
                      </span>{' '}
                    </p>

                    <p>
                      Using matched keypoints to compute the Fundamental Matrix
                      (F) that encapsulates the epipolar geometry between
                      images, employing the RANSAC algorithm to ensure accurate
                      results.
                    </p>
                    <div></div>
                  </div>
                </div>

                <div className="row">
                  <div>
                    <p>
                      <span className="head">
                        <span className="lime">3 </span> 3D Coordinate
                        Calculation
                      </span>{' '}
                    </p>

                    <p>
                      Triangulating the 3D coordinates of matched keypoints
                      using the epipolar lines from the Fundamental Matrix,
                      followed by computing the angle with respect to the real
                      coordinate axes (X, Y, Z).
                    </p>
                    <div></div>
                  </div>
                </div>

                <div className="row">
                  <div>
                    <p>
                      <span className="head">
                        <span className="lime">4 </span> Goldberg Polygon
                        Binning
                      </span>{' '}
                    </p>

                    <p>
                      Categorizing points based on their angles using the
                      Goldberg Polygon technique to create a color histogram or
                      map capturing color variations from different viewpoints
                    </p>
                    <div></div>
                  </div>
                </div>

                <div className="row">
                  <div>
                    <p>
                      <span className="head">
                        <span className="lime">5 </span>Color Data Utilization
                        for Stereo Images
                      </span>{' '}
                    </p>

                    <p>
                      Utilizing the RGB values stored in different angle bins to
                      generate stereo images or 3D reconstructions, providing a
                      detailed visualization of the scene or object.
                    </p>
                    <div></div>
                  </div>
                </div>
                <div className="row">
                  <div>
                    <p></p>
                  </div>
                </div>
              </div>
            </div>
            <div id="pf_con_con">
              <div id="pfs_container" className="short">
                <p className="pfTitle">Contributions</p>
                <div className="row">
                  <div>
                    <p>
                      <span className="head">
                        <span className="lime">1 </span>Kishore Shenoy
                      </span>
                    </p>
                    <p>
                      Contributed to the methodology of the project, and making
                      of the website.
                      <br /> <br />
                      Future contributions include generating a GT specific
                      dataset, generating the goldberg polyhedron model for
                      vertices.
                    </p>
                  </div>
                </div>
                <div className="row">
                  <div>
                    <p>
                      <span className="head">
                        <span className="lime">2 </span> Rajeshwari Devaramani
                      </span>
                    </p>
                    <p>
                      Contributed towards the methodology of the project and
                      implementation of the structure from motion algorithm.{' '}
                      <br /> <br />
                      Future contributions include vertex color generation and
                      color polling based on angle during stereo image-sequence
                      generation.
                    </p>
                  </div>
                </div>
              </div>
            </div>
            <div id="pf_con_con">
              <div id="pfs_container" className="short">
                <p className="pfTitle">References</p>
                <div className="row">
                  <div>
                    <p className="cite">
                      <span className="lime cite">1 </span> D. G. Lowe,
                      "Distinctive image features from scale-invariant
                      keypoints," *International Journal of Computer Vision*,
                      vol. 60, no. 2, pp. 91-110, 2004. [Online]. Available:{' '}
                      <a
                        href="https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf"
                        target="_blank"
                      >
                        https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
                      </a>
                      .
                    </p>
                  </div>
                </div>
                <div className="row">
                  <div>
                    <p className="cite">
                      <span className="lime cite">2 </span> NVIDIA Corporation,
                      "CUDA convolutionSeparable SDK documentation," [Online].
                      Available:{' '}
                      <a
                        href="https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_64_website/projects/convolutionSeparable/doc/convolutionSeparable.pdf"
                        target="_blank"
                      >
                        https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_64_website/projects/convolutionSeparable/doc/convolutionSeparable.pdf
                      </a>
                      .
                    </p>
                  </div>
                </div>
                <div className="row">
                  <div>
                    <p className="cite">
                      <span className="lime cite">3 </span> M. N. S. M. Omar, M.
                      F. A. Rasid, and Z. M. Zain, "Parallelization and
                      Optimization of SIFT on GPU Using CUDA," *ResearchGate*,
                      2016. [Online]. Available:{' '}
                      <a
                        href="https://www.researchgate.net/publication/269302930_Parallelization_and_Optimization_of_SIFT_on_GPU_Using_CUDA"
                        target="_blank"
                      >
                        https://www.researchgate.net/publication/269302930_Parallelization_and_Optimization_of_SIFT_on_GPU_Using_CUDA
                      </a>
                      .
                    </p>
                  </div>
                </div>
                <div className="row">
                  <div>
                    <p className="cite">
                      <span className="lime cite">4 </span> A. Minnaar,
                      "Implementing Convolutions in CUDA," 2019. [Online].
                      Available:{' '}
                      <a
                        href="https://alexminnaar.com/2019/07/12/implementing-convolutions-in-cuda.html"
                        target="_blank"
                      >
                        https://alexminnaar.com/2019/07/12/implementing-convolutions-in-cuda.html
                      </a>
                      .
                    </p>
                  </div>
                </div>
                <div className="row">
                  <div>
                    <p className="cite">
                      <span className="lime cite">5 </span> S. S. Kumar and K.
                      S. Babu, "Face Detection Using OpenCV," *Journal of Signal
                      and Information Processing*, vol. 7, no. 2, pp. 103-112,
                      2016. [Online]. Available:{' '}
                      <a
                        href="https://www.scirp.org/journal/paperinformation?paperid=73133"
                        target="_blank"
                      >
                        https://www.scirp.org/journal/paperinformation?paperid=73133
                      </a>
                      .
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </MathJaxContext>
  );
}

export default App;
