// For some weird reason we have to use this in order for the code to work in Firefox
import * as THREE from 'https://unpkg.com/three@0.150.1/build/three.module.js'
import { OrbitControls } from 'https://unpkg.com/three@0.150.1/examples/jsm/controls/OrbitControls'
import { PLYLoader } from 'https://unpkg.com/three@0.150.1/examples/jsm/loaders/PLYLoader'
// import Stats from 'three/examples/jsm/libs/stats.module'
import { supportedScenes, slugSceneMap } from 'https://data.ciirc.cvut.cz/public/projects/2023TetraNeRF/resources/scenes.js'
// import * as THREE from 'three'
// import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'
// import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader'
// // import Stats from 'three/examples/jsm/libs/stats.module'
// import { supportedScenes, slugSceneMap } from 'scenes'


function downloadVideoSlow(uri, onProgress) {
    const FPS = 60;
    return fetch(uri)
        .then(res => res.blob())
        .then(blob => {
            return new Promise((res) => {
                const fr = new FileReader();
                fr.onload = e => res(fr.result);
                fr.readAsDataURL(blob);
            })
        }).then(async (base64str) => {
            const video = document.createElement("video");
            video.src = base64str;
            video.controls = true;

            while (isNaN(video.duration))
                await new Promise((r) => setTimeout(r, 50));

            const c = document.createElement("canvas");
            Object.assign(c, {
                width: video.videoWidth,
                height: video.videoHeight,
            });
            const ctx = c.getContext("2d", { willReadFrequently: true });
            const frames = [],
                copy = () => {
                    ctx.drawImage(video, 0, 0)
                    return ctx.getImageData(0, 0, video.videoWidth, video.videoHeight);
                };
            const totalFrames = Math.ceil(video.duration * FPS);
            let currentFrames = 0;
            video.currentTime = 0;
            while (video.currentTime < video.duration) {
                video.currentTime = video.currentTime + 1 / FPS;
                await new Promise((next) => {
                    video.addEventListener('seeked', () => {
                        frames.push(copy());
                        currentFrames++;
                        if (onProgress) {
                            onProgress({
                                status: "processing",
                                total: totalFrames,
                                transferred: currentFrames,
                                percentage: currentFrames / totalFrames,
                                message: "Using slow implementation, install Google Chrome."
                            })
                        }
                        next();
                    }, {
                        once: true
                    });
                });
            }
            if (onProgress) {
                onProgress({
                    status: "processing",
                    total: frames.length,
                    transferred: frames.length,
                    percentage: 1,
                    message: "Using slow implementation, install Google Chrome."
                });
            }
            return { frames };
        });
}


function downloadVideoFast(uri, onProgress) {
    const worker = new Worker("./worker.js");
    let receivedFrames = 0;
    return new Promise((resolve, reject) => {
        try {
            const frames = [];
            // const canvas = document.querySelector("canvas").transferControlToOffscreen();
            function handleMessage(message) {
                if (message.data.error) {
                    worker.removeEventListener("message", handleMessage);
                    reject(message.data.error);
                    return
                } else if (message.data.frame) {
                    let {arrayBuffer, init} = message.data.frame;
                    const frame = new VideoFrame(arrayBuffer, init);
                    frames[message.data.i] = frame;

                    receivedFrames++;
                    if (onProgress) {
                        onProgress({
                            status: "processing",
                            transferred: receivedFrames,
                            total: message.data.total,
                            percentage: receivedFrames/message.data.total,
                        });
                    }
                    if (receivedFrames == message.data.total) {
                        worker.removeEventListener("message", handleMessage);
                        resolve({
                            frames,
                        });
                        return
                    }
                }
            }
            worker.addEventListener("message", handleMessage);
            worker.postMessage({ uri }, []);
        } catch (error) {
            reject(error);
        }
    }).then(({frames}) => Promise.all(frames).then(frames => ({frames})));
}

function downloadVideo(uri, onProgress) {
    if (!window.VideoDecoder) {
        console.error("Using SLOW firefox implementation. Please use Google Chrome");
        return downloadVideoSlow(uri, onProgress)
    } else {
        return downloadVideoFast(uri, onProgress)
    }
}


function buildDOMLoadingProgress() {
    let loadingProgress = document.createElement("div");
    loadingProgress.classList.add("loading-progress");
    loadingProgress.style.display = "none"
    loadingProgress.innerHTML = `
<div class="lp-progress-group">
    <span class="lp-label"><span class="lp-percentage">45</span><span class="lp-label-smaller">%</span></span>
    <div class="lp-pie">
      <div class="lp-left-side lp-half-circle"></div>
      <div class="lp-right-side lp-half-circle"></div>
    </div>
    <div class="lp-shadow"></div>
</div>
<div class="lp-error-group" style="display: none">
<svg viewBox="0 0 580 512">
    <g><path d="M569.517 440.013C587.975 472.007 564.806 512 527.94 512H48.054c-36.937 0-59.999-40.055-41.577-71.987L246.423 23.985c18.467-32.009 64.72-31.951 83.154 0l239.94 416.028zM288 354c-25.405 0-46 20.595-46 46s20.595 46 46 46 46-20.595 46-46-20.595-46-46-46zm-43.673-165.346l7.418 136c.347 6.364 5.609 11.346 11.982 11.346h48.546c6.373 0 11.635-4.982 11.982-11.346l7.418-136c.375-6.874-5.098-12.654-11.982-12.654h-63.383c-6.884 0-12.356 5.78-11.981 12.654z"></path></g>
</svg>
</div>
<div class="lp-status">loading</div>
`;
    const lpProgressGroup = loadingProgress.querySelector(".lp-progress-group")
    const lpErrorGroup = loadingProgress.querySelector(".lp-error-group");
    const statusElement = loadingProgress.querySelector(".lp-status");
    const fullCircleElement = loadingProgress.querySelector(".lp-pie");
    const rightSideElement = fullCircleElement.querySelector(".lp-right-side");
    const leftSideElement = fullCircleElement.querySelector(".lp-left-side");
    const percentageElement = loadingProgress.querySelector(".lp-percentage");
    let currentMode = "none";
    const setProgress = (mode, status, progress) => {
        if (mode === "none") {
            if (currentMode !== mode) {
                loadingProgress.style.display = "none";
                currentMode = mode;
            }
        } else if (mode === "error") {
            if (currentMode !== mode) {
                lpProgressGroup.style.display = "none";
                lpErrorGroup.style.removeProperty("display");
                loadingProgress.style.removeProperty("display");
                currentMode = mode;
            }

            // Update status
            statusElement.innerText = status;
        } else if (mode === "progress") {
            if (currentMode !== mode) {
                lpErrorGroup.style.display = "none";
                lpProgressGroup.style.removeProperty("display");
                loadingProgress.style.removeProperty("display");
                currentMode = mode;
            }

            // Update status
            statusElement.innerText = status;

            // Update progress
            if (isNaN(progress)) {
                progress = 0;
            }
            if (progress > 1) { progress = 1; }
            if (progress < 0) { progress = 0; }
            percentageElement.innerText = (progress * 100).toFixed(0);
            leftSideElement.style.transform = `rotate(${progress*360}deg)`
            if (progress < 0.5) {
                rightSideElement.style.display = "none";
                fullCircleElement.style.clip = "rect(0, 1em, 1em, 0.5em)";
            } else {
                rightSideElement.style.removeProperty("display");
                rightSideElement.style.transform = `rotate(180deg)`;
                fullCircleElement.style.clip = "rect(auto, auto, auto, auto)";
            }
        }   
    }
    return {loadingProgress, setProgress};
}


function buildDOM({modes, activeMode, onModeChange, domElement=null}) {
    if (!domElement) {
        domElement = document.body;
    }
    let containerElement = domElement;
    let rendererElement = document.createElement("canvas");
    containerElement.appendChild(rendererElement);
    let imageRendererElement = document.createElement("canvas");
    containerElement.appendChild(imageRendererElement);
    let controlLayer = document.createElement("div");
    containerElement.appendChild(controlLayer);
    Object.assign(imageRendererElement, {width:containerElement.clientWidth, height:containerElement.clientHeight}); 
    let controls = document.createElement("div");
    let modeButtonMap = {};
    const rebuildControls = (modes, activeMode) => {
        modeButtonMap = {};
        if (!modes) {
            modes = ["point cloud", "tetrahedra", "colour", "depth"];
        }
        controls.innerHTML = ""
        if (!activeMode) {
            activeMode = "colour";
        }
        for (const mode of modes) {
            const button = document.createElement("button");
            button.innerText = mode;
            if (mode === activeMode) {
                button.classList.add("active");
            }
            button.addEventListener("click", () => {
                if (onModeChange) {
                    onModeChange(mode);
                }
                controls.querySelectorAll("button").forEach(x => x.classList.remove("active"));
                button.classList.add("active");
            });
            controls.appendChild(button);
            modeButtonMap[mode] = button;
        }

    }
    rebuildControls(modes, activeMode);
    controls.classList.add("controls");
    const setActiveMode = (mode) => {
        controls.querySelectorAll("button").forEach(x => x.classList.remove("active"));
        modeButtonMap[mode].classList.add("active");
    };
    const {loadingProgress, setProgress} = buildDOMLoadingProgress();
    containerElement.appendChild(loadingProgress);
    containerElement.appendChild(controls);
    return {rendererElement, imageRendererElement, containerElement, controlLayer, setProgress, setActiveMode, rebuildControls};
}


class ImageRenderer {
    #imageRendererElement = null;
    #ctx = null;
    #origAngle = 0;
    #polarAngle = null;
    #distance = null;

    constructor(imageRendererElement) {
        this.#imageRendererElement = imageRendererElement;
        this.#ctx = imageRendererElement.getContext("2d");
    }

    resetScene(controls) {
        const camera = controls.object;
        this.#polarAngle = controls.getPolarAngle();
        this.#distance = controls.getDistance();
        this.#origAngle = Math.atan2(camera.position.x, camera.position.z);
    }

    activateControls(controls) {
        controls.maxPolarAngle = this.#polarAngle;
        controls.minPolarAngle = this.#polarAngle;
        controls.maxDistance = this.#distance;
        controls.minDistance = this.#distance;
    }

    deactivateControls(controls) {
        controls.maxPolarAngle = Math.PI;
        controls.minPolarAngle = 0;
        controls.maxDistance = Infinity;
        controls.minDistance = 0;
    }

    render(frames, camera) {
        const angle = (Math.atan2(camera.position.x, camera.position.z) + 4 * Math.PI - this.#origAngle) % (Math.PI * 2);
        const frameId = Math.max(0, Math.min(Math.round(angle * (frames.length - 1) / 2 / Math.PI), frames.length - 1));
        const frame = frames[frameId];
        
        const targetAspect = this.#imageRendererElement.width/this.#imageRendererElement.height;
        let sx = 0;
        let sy = 0;
        let swidth = frame.displayWidth || frame.width;
        let sheight = frame.displayHeight || frame.height;
        const sourceAspect = swidth/sheight;
        if (sourceAspect > targetAspect) {
            // Clipping horizontally
            let newwidth = targetAspect * sheight; 
            sx = (swidth - newwidth) / 2;
            swidth = newwidth;
        } else {
            // Clipping vertically
            let newheight = swidth / targetAspect; 
            sy = (sheight - newheight) / 2;
            sheight = newheight;
        }
        this.#ctx.drawImage(frame, sx, sy, swidth, sheight, 0, 0, this.#imageRendererElement.width, this.#imageRendererElement.height);
    }

    setSize(width, height) {
        Object.assign(this.#imageRendererElement, {width, height});
    }
}


class Renderer {
    #rendererElement = null;
    #imageRendererElement = null;
    #containerElement = null;
    #uiSetProgress = null;
    #controlLayer = null;
    #uiSetActiveMode = null;
    #uiRebuildControls = null;
    #currentTask = null;

    #camera = null;
    #renderer = null;
    #scene = null;
    #imageRenderer = null;
    #controls = null;

    #tetrahedraMaterial = null;
    #pointsMaterial = null;

    activeMode = null;

    #sceneData = {};
    modes = null;

    constructor(domElement, options = {}) {
        this.modes = options.modes || ["point cloud", "tetrahedra", "colour", "depth"],
        this.activeMode = options.activeMode || "colour";
        this.clearSceneData();
        let {rendererElement, imageRendererElement, containerElement, controlLayer, setProgress, setActiveMode, rebuildControls} = buildDOM({
            domElement,
            modes: this.modes,
            activeMode: this.activeMode,
            onModeChange: (mode) => {
                this.#onModeChange(mode);
            }
        });
        this.#rendererElement = rendererElement;
        this.#imageRendererElement = imageRendererElement;
        this.#containerElement = containerElement;
        this.#controlLayer = controlLayer;
        this.#uiSetProgress = setProgress;
        this.#uiSetActiveMode = setActiveMode;
        this.#uiRebuildControls = rebuildControls;
        this.initComponents();
        this.switchMode(this.activeMode);
        if(!this.modes.includes(this.activeMode)) {
            this.#uiSetActiveMode("colour");
            this.#onModeChange("colour");
        }
    }

    #onModeChange(mode) {
        window.stateChangeIsLocal = true;
        window.location.search
        const paramsSearch = new URLSearchParams(window.location.search.substr(1));
        const paramsHash = new URLSearchParams(window.location.hash.substr(1));
        if (paramsHash.has("mode")) {
            paramsHash.set("mode", mode.replace(" ", "-"));
        } else {
            paramsSearch.set("mode", mode.replace(" ", "-"));
        }
        const clonedUrl = new URL(window.location.href);
        clonedUrl.search = paramsSearch.toString();
        clonedUrl.hash = paramsHash.toString();
        window.history.replaceState({}, "", clonedUrl);
        this.switchMode(mode);
    }

    initComponents() {
        const camera = new THREE.PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        )
        camera.position.z = 40
        camera.position.x = 0.92;// 0.92914421
        camera.position.y = 0.42;
        camera.position.z = 0;
        this.#camera = camera;

        this.#renderer = new THREE.WebGLRenderer({ alpha: true, canvas: this.#rendererElement });
        this.#renderer.outputEncoding = THREE.sRGBEncoding
        this.#renderer.setSize(this.#containerElement.clientWidth, this.#containerElement.clientHeight);

        this.#scene = this.#createThreeScene();
        // Controls
        this.#controls = new OrbitControls(camera, this.#controlLayer);
        this.#controls.enableDamping = true;
        this.#controls.autoRotate = true;
        this.#controls.autoRotateSpeed = 7.5;
        this.#controls.addEventListener("end", () => {
            this.#controls.autoRotate = false;
        });

        this.#imageRenderer = new ImageRenderer(this.#imageRendererElement);
        this.#imageRenderer.resetScene(this.#controls);


        this.#tetrahedraMaterial = new THREE.MeshBasicMaterial({
            color: 0x000000,
            wireframe: true,
            opacity: 0.1,
            transparent: true,
            });

        this.#pointsMaterial = new THREE.PointsMaterial({
            vertexColors: true,
            size: 0.005,
        });

        window.addEventListener('resize', this.updateSize.bind(this));
    }

    #createThreeScene() {
        const scene = new THREE.Scene({});
        // scene.add(new THREE.AxesHelper(5))
        scene.background = new THREE.Color( 0xffffff );
        return scene;
    }

    setModes(modes) {
        if(this.modes.length === modes.length) {
            return;
        }
        this.modes = modes;
        if(!this.modes.includes(this.activeMode)) {
            console.log(`Switching modes ${this.activeMode}->colour`)
            this.#onModeChange("colour");
        }
        this.#uiRebuildControls(this.modes, this.activeMode);
    }

    updateSize() {
        if (this.#renderer === null) { return; }
        this.#imageRenderer.setSize(this.#containerElement.clientWidth, this.#containerElement.clientHeight); 
        this.#renderer.setSize(this.#containerElement.clientWidth, this.#containerElement.clientHeight); 
        const newAspect = this.#containerElement.clientWidth/this.#containerElement.clientHeight;
        if (Math.abs(this.#camera.aspect - newAspect) > 1e-6) {
            this.#camera.aspect = newAspect;
            this.#camera.updateProjectionMatrix();
        }
    }

    render() {
        this.#updateLoadingStatus();
        this.#controls.update();
        const mode = this.activeMode;
        if (mode === "tetrahedra" || mode === "point cloud") {
            if (this.#scene !== null) {
                this.#renderer.render(this.#scene, this.#camera);
            }
        } else if (mode === "colour" || mode === "depth") {
            const frames = this.#sceneData[mode]["data"];
            if (frames) {
                this.#imageRenderer.render(frames, this.#camera);
            }
        }
    }

    getDomElement() {
        return this.#containerElement;
    }

    clearSceneData() {
        this.#sceneData = {};
        if (this.#sceneData["point cloud"] && this.#sceneData["point cloud"]["data"]) {
            this.#sceneData["point cloud"]["data"].destroy();
            this.#sceneData["point cloud"]["data"] = null;
        }
        if (this.#sceneData["tetrahedra"] && this.#sceneData["tetrahedra"]["data"]) {
            this.#sceneData["tetrahedra"]["data"].destroy();
            this.#sceneData["tetrahedra"]["data"] = null;
        }
        if (this.#sceneData["colour"] && this.#sceneData["colour"]["data"]) {
            const frames = this.#sceneData["colour"]["data"];
            for (const i of frames) {
                frames[i].close();
            }
        }
        if (this.#sceneData["depth"] && this.#sceneData["depth"]["data"]) {
            const frames = this.#sceneData["depth"]["data"];
            for (const i of frames) {
                frames[i].close();
            }
        }
        for (const mode of ["point cloud", "tetrahedra", "colour", "depth"]) {
            this.#sceneData[mode] = {
                data: null,
                status: "",
            };
        }
    }

    setScene({
        tetrahedra_uri,
        color_video_uri,
        depth_video_uri,
        initial_camera,
    }) {
        const taskId = Math.floor(Math.random() * 999999999);
        this.#currentTask = taskId;
        this.clearSceneData();
        this.#scene = this.#createThreeScene();

        // Set scene camera
        this.#camera.position.x = initial_camera[0];
        this.#camera.position.y = initial_camera[1];
        this.#camera.position.z = initial_camera[2]; 

        // Download everything
        const promises = [];
        if(tetrahedra_uri) {
            const loader = new PLYLoader();
            const tetrahedarPromise = loader.load(
                tetrahedra_uri,
                (geometry) => {
                    if (this.#currentTask != taskId) {
                        geometry.dispose();
                        return;
                    }
                    geometry.computeVertexNormals()
                    this.#sceneData["tetrahedra"]["data"] = new THREE.Mesh(geometry, this.#tetrahedraMaterial);
                    this.#notifyDataAvailable("tetrahedra");
                    this.#sceneData["point cloud"]["data"] = new THREE.Points(geometry, this.#pointsMaterial);
                    this.#notifyDataAvailable("point cloud");
                },
                (xhr) => {
                    if (this.#currentTask != taskId) {
                        return;
                    }
                    // console.log((xhr.loaded / xhr.total) * 100 + '% loaded')
                    Object.assign(this.#sceneData["tetrahedra"], {
                        percentage: xhr.loaded/xhr.total,
                        status: "downloading",
                    });
                    Object.assign(this.#sceneData["point cloud"], {
                        percentage: xhr.loaded/xhr.total,
                        status: "downloading",
                    });
                    this.#updateLoadingStatus();
                },
                (error) => {
                    if (this.#currentTask != taskId) {
                        return;
                    }
                    console.error(error);
                    Object.assign(this.#sceneData["tetrahedra"], {
                        isError: true,
                        status: "failed to load",
                    });
                    Object.assign(this.#sceneData["point cloud"], {
                        isError: true,
                        status: "failed to load",
                    });
                    this.#updateLoadingStatus();
                },
            )
            promises.push(tetrahedarPromise);
        }

        if (!window.VideoDecoder) {
            Object.assign(this.#sceneData["colour"], {
                isError: true,
                status: "unsupported browser - install Google Chrome",
            });
            Object.assign(this.#sceneData["depth"], {
                isError: true,
                status: "unsupported browser - install Google Chrome",
            });
            this.#updateLoadingStatus();
        } else {
            const colorPromise = downloadVideo(color_video_uri, ({ status, percentage }) => {
                Object.assign(this.#sceneData["colour"], { percentage, status });
                this.#updateLoadingStatus();
            }).then(({ frames }) => {
                if (this.#currentTask != taskId) {
                    for (const frame of frames) {
                        frame.close();
                    }
                    return;
                }
                this.#sceneData["colour"]["data"] = frames;
                this.#notifyDataAvailable("colour");
            }).catch((error) => {
                console.error(error);
                if (this.#currentTask != taskId) {
                    return;
                }
                Object.assign(this.#sceneData["colour"], {
                    isError: true,
                    status: "failed to load",
                });
                this.#updateLoadingStatus();
            });
            promises.push(colorPromise);

            const depthPromise = downloadVideo(depth_video_uri, ({ status, percentage }) => {
                Object.assign(this.#sceneData["depth"], { percentage, status });
                this.#updateLoadingStatus();
            }).then(({ frames }) => {
                if (this.#currentTask != taskId) {
                    for (const frame of frames) {
                        frame.close();
                    }
                    return;
                }
                this.#sceneData["depth"]["data"] = frames;
                this.#notifyDataAvailable("depth");
            }).catch((error) => {
                console.error(error);
                if (this.#currentTask != taskId) {
                    return;
                }
                Object.assign(this.#sceneData["depth"], {
                    isError: true,
                    status: "failed to load",
                });
                this.#updateLoadingStatus();
            });
            promises.push(depthPromise);
        }
        return Promise.all(promises);
    }

    switchMode(mode) {
        this.#scene = null;
        if (mode === "tetrahedra") {
            this.#imageRenderer.deactivateControls(this.#controls);
            this.#imageRendererElement.style.display = "none";
            this.#rendererElement.style.display = "block";
            this.#scene = this.#createThreeScene();
            if (this.#sceneData[mode]["data"]) {
                this.#scene.add(this.#sceneData[mode]["data"]);
            }
        } else if (mode == "point cloud") {
            this.#imageRenderer.deactivateControls(this.#controls);
            this.#imageRendererElement.style.display = "none";
            this.#rendererElement.style.display = "block";
            this.#scene = this.#createThreeScene();
            if (this.#sceneData[mode]["data"]) {
                this.#scene.add(this.#sceneData[mode]["data"]);
            }
        } else if (mode === "colour") {
            this.#imageRenderer.activateControls(this.#controls);
            this.#rendererElement.style.display = "none";
            this.#imageRendererElement.style.display = "block";
        } else if (mode === "depth") {
            this.#imageRenderer.activateControls(this.#controls);
            this.#rendererElement.style.display = "none";
            this.#imageRendererElement.style.display = "block";
        }
        this.activeMode = mode;
        this.#updateLoadingStatus();
    }

    setActiveMode(mode) {
        this.switchMode(mode);
        this.#uiSetActiveMode(mode);
    }

    #notifyDataAvailable(mode) {
        if (this.activeMode === mode) {
            this.switchMode(this.activeMode);
        }
        this.#updateLoadingStatus();
    }

    #updateLoadingStatus() {
        let mode = "none";
        let status = "";
        let progress = 0;

        if (this.#sceneData[this.activeMode].data) {
            mode = "none";
        } else if (this.#sceneData[this.activeMode].isError) {
            mode = "error";
            status = this.#sceneData[this.activeMode]["status"];
        } else {
            mode = "progress";
            status = this.#sceneData[this.activeMode]["status"];
            progress = this.#sceneData[this.activeMode]["percentage"];
            if (isNaN(progress)) {
                progress = 0;
            }
        }
        this.#uiSetProgress(mode, status, progress);
    }

    #animate() {
        requestAnimationFrame(this.#animate.bind(this));
        this.render();
    }

    start() {
        this.#animate();
    }
}

// Read current scene from the slug
function getCurrentData() {
    const paramsSearch = new URLSearchParams(window.location.search.substr(1));
    const paramsHash = new URLSearchParams(window.location.hash.substr(1));
    const params = new URLSearchParams({
        ...Object.fromEntries(paramsSearch),
        ...Object.fromEntries(paramsHash)
    });
    let sceneData = supportedScenes[0]; // Default
    let mode = "colour";  // Default
    if (params.has("scene") && slugSceneMap[params.get("scene")]) {
        sceneData = supportedScenes[slugSceneMap[params.get("scene")]];
    }
    if (params.has("mode")) {
        mode = params.get("mode").replace("-", " ");
    }
    return {
        sceneData,
        mode,
    };
}

// Load scene from href
const initialData = getCurrentData();
let currentSceneSlug = initialData.sceneData.slug;
console.log(`Loading initial scene ${currentSceneSlug}`);
const rendererDomElement = document.getElementById("viewer-port");
let size = initialData.sceneData.size;
Object.assign(rendererDomElement.style, {
    "aspect-ratio": `${size[0]}/${size[1]}`,
    "max-width": `min(100%, ${size[0]}px)`,
    "max-height": `${size[1]}px`,
});
const renderer = new Renderer(rendererDomElement, {
    activeMode: initialData.mode,
    modes: initialData.sceneData.modes,
});
renderer.setScene(initialData.sceneData);
renderer.updateSize();
function animate() {
    requestAnimationFrame(animate);
    renderer.render();
}
animate();

addEventListener("hashchange", (event) => {
    const newUrl = (new URL(event.newURL)).hash;
    const newDataSearch = new URLSearchParams(newUrl.search.substr(1));
    const newDataHash = new URLSearchParams(newUrl.hash.substr(1));
    let newData = new URLSearchParams({
        ...Object.fromEntries(newDataSearch),
        ...Object.fromEntries(newDataHash)
    });
    if (renderer.activeMode !== newData.mode) {
        renderer.setActiveMode(newData.mode);
    }
    if (newData.sceneData.slug !== currentSceneSlug) {
        console.log(`Changing scene to ${newData.sceneData.name} [${newData.sceneData.modes}]`);
        renderer.setModes(newData.sceneData.modes);
        renderer.setScene(newData.sceneData);
        let size = newData.sceneData.size;
        Object.assign(rendererDomElement.style, {
            "aspect-ratio": `${size[0]}/${size[1]}`,
            "max-width": `min(100%, ${size[0]})`,
            "max-height": `${size[1]}px`,
        });
        renderer.updateSize();
        currentSceneSlug = newData.sceneData.slug;
    }
});
