importScripts("https://gpac.github.io/mp4box.js/dist/mp4box.all.js");


class MP4FileSink {
    #setProgress = null;
    #file = null;
    #offset = 0;
    #totalSize = undefined;
  
    constructor(file, totalSize, setProgress) {
        this.#file = file;
        this.#setProgress = setProgress.bind(this);
        this.#totalSize = totalSize;
    }
  
    write(chunk) {
        // MP4Box.js requires buffers to be ArrayBuffers, but we have a Uint8Array.
        const buffer = new ArrayBuffer(chunk.byteLength);
        new Uint8Array(buffer).set(chunk);

        // Inform MP4Box where in the file this chunk is from.
        buffer.fileStart = this.#offset;
        this.#offset += buffer.byteLength;

        // Append chunk.
        this.#setProgress({
            status: "download",
            transferred: this.#offset,
            total: this.#totalSize,
            percentage: this.#offset/this.#totalSize,
        });
        this.#file.appendBuffer(buffer);
    }
  
    close() {
        this.#setProgress({
            status: "download",
            transferred: this.#totalSize,
            total: this.#totalSize,
            percentage: 1,
        });
        this.#file.flush();
    }
}


async function safeClone(data) {
    const rect = {x:0, y:0, width:data.codedWidth, height:data.codedHeight};
    const arrayBuffer = new Uint8Array(data.allocationSize({rect}));
    await data.copyTo(arrayBuffer, {rect});

    const init = {
        codedHeight: data.codedHeight,
        codedWidth: data.codedWidth,
        displayHeight: data.displayHeight,
        displayWidth: data.displayWidth,
        duration: data.duration || undefined,
        format: data.format,
        timestamp: data.timestamp,
        visibleRect: data.visibleRect || undefined
    };

    return {arrayBuffer, init};
    const frame = new VideoFrame(arrayBuffer, init);
}
  

// Startup.
function start({uri}) {
    let processedFrames = 0;
    let totalFrames = 0;

    function onProgress(progress) {
        self.postMessage({progress});
    }

    const decoder = new VideoDecoder({
        async output(oldFrame) {
            const frameIndex = processedFrames;
            processedFrames++;
            const frame = await safeClone(oldFrame);
            oldFrame.close()
            self.postMessage({
                frame,
                i: frameIndex,
                total: totalFrames,
            });
        },
        error(e) {
            self.postMessage({
                error: e,
            });
        },
    });

    // Configure an MP4Box File for demuxing.
    const file = MP4Box.createFile();
    file.onError = error => setStatus("demux", error);
    file.onReady = (info) => {
        // Generate and emit an appropriate VideoDecoderConfig.
        const track = info.videoTracks[0];
        const trak = file.getTrackById(track.id);
        totalFrames = Math.max(track.nb_samples, totalFrames);
        let description = null;
        for (const entry of trak.mdia.minf.stbl.stsd.entries) {
            if (entry.avcC || entry.hvcC) {
                const stream = new DataStream(undefined, 0, DataStream.BIG_ENDIAN);
                if (entry.avcC) {
                    entry.avcC.write(stream);
                } else {
                    entry.hvcC.write(stream);
                }
                description = new Uint8Array(stream.buffer, 8);  // Remove the box header.
                break;
            }
        }
        if (description === null) {
            throw "avcC or hvcC not found";
        }

        decoder.configure({
            codec: track.codec,
            codedHeight: track.video.height,
            codedWidth: track.video.width,
            description: description
        });

        // Start demuxing.
        file.setExtractionOptions(track.id);
        file.start();
    }
    file.onSamples = (track_id, ref, samples) => {
        // Generate and emit an EncodedVideoChunk for each demuxed sample.
        for (const sample of samples) {
            decoder.decode(new EncodedVideoChunk({
                type: sample.is_sync ? "key" : "delta",
                timestamp: 1e6 * sample.cts / sample.timescale,
                duration: 1e6 * sample.duration / sample.timescale,
                data: sample.data
            }));
        }
        return decoder.flush();
    }
    
    // Fetch the file and pipe the data through.
    return fetch(uri).then(response => {
        // highWaterMark should be large enough for smooth streaming, but lower is
        // better for memory usage.
        const totalSize = Number(response.headers.get("content-length"))
        const fileSink = new MP4FileSink(file, totalSize, onProgress);
        response.body.pipeTo(new WritableStream(fileSink, {highWaterMark: 2}));
    }).catch((error) => {
        self.postMessage({ error });
    });
}

// Listen for the start request.
self.addEventListener("message", message => start(message.data), {once: true});