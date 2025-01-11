function updateBrushSettings() {
    const brushSize = document.getElementById("brushSize").value;
    const brushColor = document.getElementById("brushColor").value;

    fetch("/update_settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ brush_size: brushSize, brush_color: brushColor }),
    }).then(response => console.log("Brush settings updated"));
}

function resetCanvas() {
    fetch("/reset_canvas", { method: "POST" }).then(response => console.log("Canvas reset"));
}

function undo() {
    fetch("/undo", { method: "POST" }).then(response => console.log("Undo action performed"));
}

function redo() {
    fetch("/redo", { method: "POST" }).then(response => console.log("Redo action performed"));
}

function saveCanvas() {
    fetch("/save_canvas", { method: "GET" })
        .then(response => response.json())
        .then(data => {
            if (data.status === "saved") {
                const link = document.createElement("a");
                link.href = `/${data.filename}`;
                link.download = "drawing.png";
                link.click();
                console.log("Canvas saved:", data.filename);
            } else {
                alert(data.message);
            }
        });
}
