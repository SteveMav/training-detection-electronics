from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from tkinter import BOTH, LEFT, RIGHT, VERTICAL, DoubleVar, IntVar, StringVar, Tk, filedialog, messagebox
from tkinter import ttk

from PIL import Image, ImageDraw, ImageFont, ImageTk


DEFAULT_MODEL = Path("models") / "electrocom61" / "best.pt"
IMAGE_FILE_TYPES = (
    ("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
    ("All files", "*.*"),
)


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]


class DetectorApp(Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("ElectroCom-61 Tester")
        self.geometry("1180x760")
        self.minsize(980, 620)

        self.image_path: Path | None = None
        self.annotated_image: Image.Image | None = None
        self.preview_image: Image.Image | None = None
        self.tk_preview: ImageTk.PhotoImage | None = None
        self.model_cache: dict[str, object] = {}

        self.model_var = StringVar(value=str(DEFAULT_MODEL))
        self.conf_var = DoubleVar(value=0.25)
        self.imgsz_var = IntVar(value=640)
        self.device_var = StringVar(value="auto")
        self.status_var = StringVar(value="Pret. Choisis une image pour tester le modele.")

        self._configure_style()
        self._build_layout()

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#f6f7f9")
        style.configure("Panel.TFrame", background="#ffffff")
        style.configure("Title.TLabel", background="#ffffff", font=("Segoe UI", 13, "bold"))
        style.configure("Muted.TLabel", background="#ffffff", foreground="#5f6b7a")
        style.configure("Status.TLabel", background="#eef2f6", foreground="#2d3748")
        style.configure("TButton", padding=(10, 7), font=("Segoe UI", 10))

    def _build_layout(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill=BOTH, expand=True)

        body = ttk.Frame(root)
        body.pack(fill=BOTH, expand=True)

        sidebar = ttk.Frame(body, style="Panel.TFrame", padding=14, width=330)
        sidebar.pack(side=LEFT, fill="y", padx=(0, 12))
        sidebar.pack_propagate(False)

        content = ttk.Frame(body, style="Panel.TFrame", padding=14)
        content.pack(side=RIGHT, fill=BOTH, expand=True)

        ttk.Label(sidebar, text="Test du modele", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            sidebar,
            text="Charge une photo, lance la detection, puis sauvegarde l'image annotee.",
            style="Muted.TLabel",
            wraplength=285,
        ).pack(anchor="w", pady=(4, 16))

        ttk.Label(sidebar, text="Modele", style="Muted.TLabel").pack(anchor="w")
        model_row = ttk.Frame(sidebar, style="Panel.TFrame")
        model_row.pack(fill="x", pady=(4, 12))
        ttk.Entry(model_row, textvariable=self.model_var).pack(side=LEFT, fill="x", expand=True)
        ttk.Button(model_row, text="Parcourir", command=self.browse_model).pack(side=RIGHT, padx=(8, 0))

        ttk.Button(sidebar, text="Ouvrir une image", command=self.open_image).pack(fill="x", pady=(0, 8))
        self.detect_button = ttk.Button(sidebar, text="Detecter", command=self.detect)
        self.detect_button.pack(fill="x", pady=(0, 8))
        self.save_button = ttk.Button(sidebar, text="Sauvegarder resultat", command=self.save_result, state="disabled")
        self.save_button.pack(fill="x")

        ttk.Separator(sidebar).pack(fill="x", pady=14)

        settings = ttk.Frame(sidebar, style="Panel.TFrame")
        settings.pack(fill="x")
        ttk.Label(settings, text="Confiance", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(settings, textvariable=self.conf_var, style="Muted.TLabel").grid(row=0, column=1, sticky="e")
        ttk.Scale(
            settings,
            from_=0.05,
            to=0.90,
            variable=self.conf_var,
        ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 12))

        ttk.Label(settings, text="Image size", style="Muted.TLabel").grid(row=2, column=0, sticky="w")
        ttk.Spinbox(
            settings,
            from_=320,
            to=1280,
            increment=64,
            textvariable=self.imgsz_var,
            width=8,
        ).grid(row=2, column=1, sticky="e")

        ttk.Label(settings, text="Device", style="Muted.TLabel").grid(row=3, column=0, sticky="w", pady=(12, 0))
        ttk.Combobox(
            settings,
            textvariable=self.device_var,
            values=("auto", "cpu", "0"),
            width=8,
        ).grid(row=3, column=1, sticky="e", pady=(12, 0))
        settings.columnconfigure(0, weight=1)

        ttk.Separator(sidebar).pack(fill="x", pady=14)
        self.file_label = ttk.Label(sidebar, text="Aucune image", style="Muted.TLabel", wraplength=285)
        self.file_label.pack(anchor="w")

        header = ttk.Frame(content, style="Panel.TFrame")
        header.pack(fill="x", pady=(0, 10))
        ttk.Label(header, text="Apercu", style="Title.TLabel").pack(side=LEFT)

        self.preview_frame = ttk.Frame(content, style="Panel.TFrame")
        self.preview_frame.pack(fill=BOTH, expand=True)
        self.preview_frame.bind("<Configure>", self._refresh_preview)

        self.preview_label = ttk.Label(
            self.preview_frame,
            text="Ouvre une image pour commencer.",
            anchor="center",
            background="#eef2f6",
            foreground="#526071",
        )
        self.preview_label.pack(fill=BOTH, expand=True)

        results_frame = ttk.Frame(content, style="Panel.TFrame")
        results_frame.pack(fill="x", pady=(12, 0))
        ttk.Label(results_frame, text="Detections", style="Title.TLabel").pack(anchor="w")

        table_frame = ttk.Frame(results_frame, style="Panel.TFrame")
        table_frame.pack(fill="x", pady=(8, 0))
        columns = ("label", "confidence", "box")
        self.results_table = ttk.Treeview(table_frame, columns=columns, show="headings", height=6)
        self.results_table.heading("label", text="Classe")
        self.results_table.heading("confidence", text="Confiance")
        self.results_table.heading("box", text="Box")
        self.results_table.column("label", width=240, anchor="w")
        self.results_table.column("confidence", width=100, anchor="center")
        self.results_table.column("box", width=260, anchor="w")
        scrollbar = ttk.Scrollbar(table_frame, orient=VERTICAL, command=self.results_table.yview)
        self.results_table.configure(yscrollcommand=scrollbar.set)
        self.results_table.pack(side=LEFT, fill="x", expand=True)
        scrollbar.pack(side=RIGHT, fill="y")

        status = ttk.Label(root, textvariable=self.status_var, style="Status.TLabel", padding=(10, 6))
        status.pack(fill="x", side="bottom", pady=(10, 0))

    def browse_model(self) -> None:
        filename = filedialog.askopenfilename(
            title="Choisir un poids YOLO",
            filetypes=(("YOLO weights", "*.pt"), ("All files", "*.*")),
        )
        if filename:
            self.model_var.set(filename)

    def open_image(self) -> None:
        filename = filedialog.askopenfilename(title="Choisir une image", filetypes=IMAGE_FILE_TYPES)
        if not filename:
            return

        self.image_path = Path(filename)
        self.annotated_image = None
        self.save_button.configure(state="disabled")
        self.file_label.configure(text=self.image_path.name)
        self._clear_results()

        image = Image.open(self.image_path).convert("RGB")
        self._set_preview(image)
        self.status_var.set(f"Image chargee: {self.image_path.name}")

    def detect(self) -> None:
        if self.image_path is None:
            messagebox.showinfo("Image manquante", "Choisis une image d'abord.")
            return

        model_path = Path(self.model_var.get())
        if not model_path.exists():
            messagebox.showerror("Modele manquant", f"Modele introuvable:\n{model_path}")
            return

        self.detect_button.configure(state="disabled")
        self.save_button.configure(state="disabled")
        self._clear_results()
        self.status_var.set("Detection en cours...")

        thread = threading.Thread(
            target=self._detect_worker,
            args=(self.image_path, model_path, float(self.conf_var.get()), int(self.imgsz_var.get()), self.device_var.get()),
            daemon=True,
        )
        thread.start()

    def _detect_worker(self, image_path: Path, model_path: Path, conf: float, imgsz: int, device: str) -> None:
        try:
            detections = self._predict(image_path, model_path, conf, imgsz, device)
            annotated = render_detections(image_path, detections)
        except Exception as exc:  # UI safety net for model/runtime errors.
            self.after(0, self._detection_failed, str(exc))
        else:
            self.after(0, self._detection_done, detections, annotated)

    def _predict(self, image_path: Path, model_path: Path, conf: float, imgsz: int, device: str) -> list[Detection]:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics n'est pas installe. Lance: .\\start_app.ps1") from exc

        model_key = str(model_path.resolve())
        model = self.model_cache.get(model_key)
        if model is None:
            model = YOLO(model_key)
            self.model_cache[model_key] = model

        predict_args: dict[str, object] = {
            "source": str(image_path),
            "conf": max(0.01, min(conf, 0.99)),
            "imgsz": max(160, imgsz),
            "verbose": False,
        }
        if device.lower() != "auto":
            predict_args["device"] = device

        results = model.predict(**predict_args)
        if not results:
            return []

        return detections_from_result(results[0])

    def _detection_done(self, detections: list[Detection], annotated: Image.Image) -> None:
        self.detect_button.configure(state="normal")
        self.annotated_image = annotated
        self._set_preview(annotated)
        self._fill_results(detections)
        self.save_button.configure(state="normal")
        self.status_var.set(f"Detection terminee: {len(detections)} objet(s).")

    def _detection_failed(self, message: str) -> None:
        self.detect_button.configure(state="normal")
        self.status_var.set(message)
        messagebox.showerror("Erreur detection", message)

    def save_result(self) -> None:
        if self.annotated_image is None or self.image_path is None:
            return

        default_dir = Path("data") / "output"
        default_dir.mkdir(parents=True, exist_ok=True)
        default_name = f"{self.image_path.stem}_detected.png"
        filename = filedialog.asksaveasfilename(
            title="Sauvegarder resultat",
            initialdir=str(default_dir.resolve()),
            initialfile=default_name,
            defaultextension=".png",
            filetypes=(("PNG image", "*.png"), ("JPEG image", "*.jpg"), ("All files", "*.*")),
        )
        if not filename:
            return

        self.annotated_image.save(filename)
        self.status_var.set(f"Resultat sauvegarde: {Path(filename).name}")

    def _set_preview(self, image: Image.Image) -> None:
        self.preview_image = image
        self._refresh_preview()

    def _refresh_preview(self, _event: object | None = None) -> None:
        if self.preview_image is None:
            return

        width = max(320, self.preview_frame.winfo_width() - 24)
        height = max(260, self.preview_frame.winfo_height() - 24)
        preview = self.preview_image.copy()
        preview.thumbnail((width, height), Image.Resampling.LANCZOS)
        self.tk_preview = ImageTk.PhotoImage(preview)
        self.preview_label.configure(image=self.tk_preview, text="", background="#111827")

    def _clear_results(self) -> None:
        for item in self.results_table.get_children():
            self.results_table.delete(item)

    def _fill_results(self, detections: list[Detection]) -> None:
        self._clear_results()
        for detection in detections:
            self.results_table.insert(
                "",
                "end",
                values=(
                    detection.label,
                    f"{detection.confidence:.2f}",
                    f"{detection.box[0]}, {detection.box[1]}, {detection.box[2]}, {detection.box[3]}",
                ),
            )


def detections_from_result(result: object) -> list[Detection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    names = getattr(result, "names", {})
    detections: list[Detection] = []

    for box in boxes:
        class_id = int(box.cls[0])
        xyxy = box.xyxy[0].tolist()
        detections.append(
            Detection(
                label=class_name(names, class_id),
                confidence=float(box.conf[0]),
                box=tuple(round(value) for value in xyxy),
            )
        )

    return detections


def render_detections(image_path: Path, detections: list[Detection]) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for detection in detections:
        x1, y1, x2, y2 = detection.box
        color = color_for_label(detection.label)
        label = f"{detection.label} {detection.confidence:.2f}"

        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        text_box = draw.textbbox((x1, y1), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_y = max(0, y1 - text_height - 6)

        draw.rectangle((x1, label_y, x1 + text_width + 8, label_y + text_height + 6), fill=color)
        draw.text((x1 + 4, label_y + 3), label, fill="white", font=font)

    return image


def class_name(names: object, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def color_for_label(label: str) -> tuple[int, int, int]:
    palette = [
        (40, 121, 255),
        (12, 166, 120),
        (232, 84, 60),
        (145, 88, 255),
        (245, 157, 35),
        (42, 161, 189),
        (196, 64, 128),
    ]
    return palette[sum(label.encode("utf-8")) % len(palette)]


def main() -> None:
    app = DetectorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
