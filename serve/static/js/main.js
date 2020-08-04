window.onload = () => {
  const reset = document.getElementById("reset");
  const predict = document.getElementById("predict");
  const picture = document.getElementById("picture");
  const canvas = document.getElementById("canvas");
  const predicted = document.getElementsByClassName("predicted");
  const picCtx = picture.getContext("2d");
  const ctx = canvas.getContext("2d");
  const lineWidth = 45;
  const lineColor = "#000000";
  const canvasWidth = 641; // 20 * 28 + 1
  const canvasHeight = 641; // 20 * 28 + 1
  const pictureWidth = 320; // 10 * 28
  const pictureHeight = 320; // 10 * 28

  let isDrawing = false;
  let curPos;
  let p_data;
  canvas.width = canvasWidth;
  canvas.height = canvasHeight;
  picture.width = pictureWidth;
  picture.height = pictureHeight;

  clear();

  function getPosition(clientX, clientY) {
    let box = canvas.getBoundingClientRect();
    return { x: clientX - box.x, y: clientY - box.y };
  }

  function draw(e) {
    if (isDrawing) {
      let pos = getPosition(e.clientX, e.clientY);
      ctx.strokeStyle = lineColor;
      ctx.lineWidth = lineWidth;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath();
      ctx.moveTo(curPos.x, curPos.y);
      ctx.lineTo(pos.x, pos.y);
      ctx.stroke();
      ctx.closePath();
      curPos = pos;
    }
  }

  canvas.onmousedown = function (e) {
    isDrawing = true;
    curPos = getPosition(e.clientX, e.clientY);
    draw(e);
  };

  canvas.onmousemove = function (e) {
    draw(e);
  };

  canvas.onmouseup = function (e) {
    isDrawing = false;

    const img = new Image();
    img.src = canvas.toDataURL();
    img.onload = function () {
      let inputs = [];
      const input = document.createElement("canvas").getContext("2d");
      /* Map the original data size to a size of 28*28 (28 * 28 = 784) */
      input.drawImage(img, 0, 0, img.width, img.height, 0, 0, 32, 32);
      let data = input.getImageData(0, 0, 32, 32).data;
      for (let i = 0; i < 32; ++i) {
        for (let j = 0; j < 32; ++j) {
          let px = 4 * (i * 32 + j);
          let r = data[px];
          let g = data[px + 1];
          let b = data[px + 2];
          inputs[i * 32 + j] = (r + g + b) / 3;
          /* Map the pixels of canvas `input` to canvas `picture` */
          picCtx.fillStyle = "rgb(" + [r, g, b].join(",") + ")";
          picCtx.fillRect(j * 10, i * 10, 10, 10);
        }
      }
      p_data = inputs;
    };
  };

  predict.onclick = function () {
    $.ajax({
      url: "/predict",
      method: "POST",
      contentType: "application/json",
      data: JSON.stringify(p_data),
      success: (ret) => {
          console.log(ret.data);
          predicted[0].innerHTML = "LeNet : " + ret.data[0];
          predicted[1].innerHTML = "ResNet : " + ret.data[1];
      },
    });
  };

  reset.onclick = clear;

  function clear() {
    ctx.fillStyle = "#ffffff";
    picCtx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);
    picCtx.fillRect(0, 0, picture.width, picture.height);

    $(".result td").text("").removeClass("answer");
  }
};

let bengalichar_codes = []
