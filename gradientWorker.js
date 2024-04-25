// self.importScripts("//pagecdn.io/lib/mathjs/7.1.0/math.min.js");

const distances = (position, points) => {
    let dist = new Array(points.length);
    for (let i in points) {
        dist[i] = distance(position, points);
    }
    return dist;
};
const distance = (position, point) => {
    return Math.sqrt(
        Math.pow(point.x - position.x, 2) + Math.pow(point.y - position.y, 2)
    );
};
const computeTime = {
    invSum: 0,
    rgb: 0,
    hue: 0,
    hsv: 0,
};
const rgbToHue = (rgb) => {
    // Using the method provided at https://en.wikipedia.org/wiki/HSL_and_HSV#Hue_and_chroma
    var M,
        m,
        C,
        hue = 0,
        c;
    M = Math.max(...rgb);
    m = Math.min(...rgb);
    C = M - m;
    if (C === 0) {
    } else if (M === rgb[0]) {
        hue = ((rgb[1] - rgb[2]) / C) % 6;
    } else if (M === rgb[1]) {
        hue = (rgb[2] - rgb[0]) / C + 2;
    } else if (M === rgb[2]) {
        hue = (rgb[0] - rgb[1]) / C + 4;
    }
    hue = (((hue * 60) % 360) + 360) % 360;
    return hue;
    return { hue: hue, M: M, m: m, C: C };
};
const hsvToRgb = (hsv) => {
    // console.log("converting", hsv)
    // Using the alternative method provided at https://en.wikipedia.org/wiki/HSL_and_HSV#HSV_to_RGB_alternative
    const f = (n) => {
        let k = (n + hsv[0] / 60) % 6;
        let output =
            hsv[2] - hsv[2] * hsv[1] * Math.max(0, Math.min(k, 4 - k, 1));
        return output;
    };
    let rgb = [f(5) * 255, f(3) * 255, f(1) * 255];
    // console.log("rgb is",rgb)
    // if(rgb[0]===0){
    //     console.log(rgb)
    // }
    return rgb;
};
const getColor = (position, points, log) => {
    let invSum = 0;
    let hsv = [0, 0, 0],
        rgb = [0, 0, 0]; //, 255]
    var t1, t0, hue;
    // t0 = Date.now();
    let colour;
    for (let i in points) {
        let invD = 1 / distance(position, points[i]);
        invSum += invD;
        colour = points[i].colour;
        hsv[1] += invD * colour.hsv.s;
        hsv[2] += invD * colour.hsv.v;

        rgb[0] += invD * colour.rgb.r;
        rgb[1] += invD * colour.rgb.g;
        rgb[2] += invD * colour.rgb.b;
    }
    // t1 = Date.now() - t0;
    // computeTime.invSum += t1;

    // t0 = Date.now();
    rgb[0] /= invSum;
    rgb[1] /= invSum;
    rgb[2] /= invSum;
    // t1 = Date.now() - t0;
    // computeTime.rgb += t1;

    // t0 = Date.now();
    hue = rgbToHue(rgb);
    // t1 = Date.now() - t0;
    // computeTime.hue += t1;

    // t0 = Date.now();
    // hsv[0] = hue.hue;
    hsv[0] = hue;
    hsv[1] /= invSum;
    hsv[2] /= invSum;
    // t1 = Date.now() - t0;
    // computeTime.hsv += t1;

    return hsvToRgb(hsv);
    return { pixel: hsvToRgb(hsv), hsv: hsv, rgb: rgb, hue: hue };
};
onmessage = (e) => {
    let imageData = e.data.imageData,
        points = e.data.points,
        canvas = e.data.canvas;
    console.log(points);
    console.log(imageData);
    if (points) {
        // var t0 = Date.now();
        const data = imageData.data;
        for (var i = 0; i < data.length; i += 4) {
            var x = Math.floor(i / 4) % canvas.width;
            var y = Math.floor(Math.floor(i / 4) / canvas.width);
            let pixel = getColor({ x: x, y: y }, points, i);
            // if (x === 10 && y === 10) {
            //     console.log("at [" + [x, y] + "]", pixel);
            //     // console.log(pixel)
            // }
            data[i] = pixel[0];
            data[i + 1] = pixel[1];
            data[i + 2] = pixel[2];
            data[i + 3] = 255;
        }
        // var t1 = Date.now() - t0;
        // console.log("invSum time elapsed=", computeTime.invSum);
        // console.log("rgb time elapsed=", computeTime.rgb);
        // console.log("hue time elapsed=", computeTime.hue);
        // console.log("hsv time elapsed=", computeTime.hsv);
        // console.log("computation time " + t1 + "ms");
        postMessage({ imageData });
    }
};
