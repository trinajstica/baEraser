/**
 * baEraser - GNOME Object Eraser
 * GTK4 + libadwaita application for removing objects from images
 * using OpenCV inpainting (classical + DNN/LaMa)
 */

#include <adwaita.h>
#include <gtk/gtk.h>
#include <gdk/gdk.h>
#include <glib.h>
#include <pango/pango.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/imgcodecs.hpp>
#include <onnxruntime_cxx_api.h>

#include <string>
#include <vector>
#include <memory>
#include <cmath>
#include <random>
#include <filesystem>
#include <thread>
#include <atomic>
#include <unistd.h>

// ─────────────────────────────────────────────
// Forward declarations
// ─────────────────────────────────────────────
struct AppState;
static void on_open_clicked(GtkButton *btn, gpointer user_data);
static void on_save_clicked(GtkButton *btn, gpointer user_data);
static void on_erase_clicked(GtkButton *btn, gpointer user_data);
static void on_undo_clicked(GtkButton *btn, gpointer user_data);
static void on_redo_clicked(GtkButton *btn, gpointer user_data);
static void on_clear_mask_clicked(GtkButton *btn, gpointer user_data);
static void on_fixit_clicked(GtkButton *btn, gpointer user_data);
static void on_pick_source_clicked(GtkButton *btn, gpointer user_data);
static void on_clone_here_clicked(GtkButton *btn, gpointer user_data);
static void on_cancel_clone_clicked(GtkButton *btn, gpointer user_data);
static void finish_pick_source(AppState *app);
static void finish_pick_destination(AppState *app);
static void on_tool_brush_clicked(GtkButton *btn, gpointer user_data);
static void on_tool_rect_clicked(GtkButton *btn, gpointer user_data);
static void on_zoom_in_clicked(GtkButton *btn, gpointer user_data);
static void on_zoom_out_clicked(GtkButton *btn, gpointer user_data);
static void on_zoom_fit_clicked(GtkButton *btn, gpointer user_data);
static void canvas_draw(GtkDrawingArea *da, cairo_t *cr, int w, int h, gpointer user_data);
static void on_draw_drag_begin(GtkGestureDrag *gesture, double x, double y, gpointer user_data);
static void on_draw_drag_update(GtkGestureDrag *gesture, double dx, double dy, gpointer user_data);
static void on_draw_drag_end(GtkGestureDrag *gesture, double dx, double dy, gpointer user_data);
static void on_pan_drag_begin(GtkGestureDrag *gesture, double x, double y, gpointer user_data);
static void on_pan_drag_update(GtkGestureDrag *gesture, double dx, double dy, gpointer user_data);
static void on_pan_drag_end(GtkGestureDrag *gesture, double dx, double dy, gpointer user_data);
static gboolean on_scroll(GtkEventControllerScroll *ctrl, double dx, double dy, gpointer user_data);
static void on_motion(GtkEventControllerMotion *ctrl, double x, double y, gpointer user_data);
static void on_motion_leave(GtkEventControllerMotion *ctrl, gpointer user_data);
static gboolean on_key_released(GtkEventControllerKey *ctrl,
                                guint keyval, guint keycode,
                                GdkModifierType state, gpointer user_data);
static gboolean drop_target_drop(GtkDropTarget *target, const GValue *value, double x, double y, gpointer user_data);
static void load_image_from_path(AppState *app, const std::string &path);
static void update_canvas_size(AppState *app);
static void update_button_states(AppState *app);
static void show_error_dialog(AppState *app, const std::string &message);
static void show_info_dialog(AppState *app, const std::string &message);
static void on_uri_launch_done(GObject *source_object, GAsyncResult *result, gpointer user_data);
static cv::Mat gdk_pixbuf_to_mat(GdkPixbuf *pixbuf);
static gboolean fit_image_idle(gpointer user_data);
static void rebuild_image_surface(AppState *app);
static void rebuild_mask_surface(AppState *app);
static void update_clone_ui(AppState *app);
static void update_canvas_cursor(AppState *app);
static void finish_mask_change(AppState *app);
static void cancel_mask_move(AppState *app);
static void set_mask(AppState *app, cv::Mat mask, bool clear_clone_destination = false);
static void clear_mask(AppState *app, bool clear_clone_destination = true);
static void register_project_icons();

// ─────────────────────────────────────────────
// Enums
// ─────────────────────────────────────────────
enum class DrawTool { BRUSH, RECT };
// App interaction mode — normal drawing vs. guided clone-heal selection
enum class AppMode  { NORMAL, PICK_SOURCE, PICK_DEST };
enum class InpaintMethod {
    TELEA,         // 0 - OpenCV Telea, fast
    NS,            // 1 - Navier-Stokes, smooth
    MULTISCALE,    // 2 - Multi-scale TELEA for larger regions
    PATCHMATCH,    // 3 - PatchMatch fallback without a model
    DNN_LAMA,      // 4 - LaMa AI
};

// ─────────────────────────────────────────────
// Application State
// ─────────────────────────────────────────────
struct AppState {
    // GTK widgets
    AdwApplicationWindow *window = nullptr;
    GtkDrawingArea       *canvas = nullptr;
    GtkWidget            *scrolled_window = nullptr;
    GtkButton            *btn_open = nullptr;
    GtkButton            *btn_save = nullptr;
    GtkButton            *btn_erase = nullptr;
    GtkButton            *btn_undo = nullptr;
    GtkButton            *btn_redo = nullptr;
    GtkButton            *btn_clear_mask = nullptr;
    GtkButton            *btn_fixit = nullptr;
    GtkButton            *btn_pick_source   = nullptr;
    GtkButton            *btn_clone_here    = nullptr;
    GtkButton            *btn_cancel_clone  = nullptr;
    GtkToggleButton      *btn_tool_brush = nullptr;
    GtkToggleButton      *btn_tool_rect = nullptr;
    GtkWidget            *spin_brush_size = nullptr;
    GtkDropDown          *method_dropdown = nullptr;
    GtkLabel             *status_label = nullptr;
    GtkLabel             *zoom_label = nullptr;
    GtkLabel             *lbl_clone_source = nullptr;
    GtkLabel             *lbl_clone_dest = nullptr;
    GtkLabel             *lbl_clone_hint = nullptr;
    GtkSpinner           *spinner = nullptr;

    // Async erase state
    std::thread          erase_thread;
    std::atomic<bool>    is_erasing{false};

    // Image data
    cv::Mat original_image;   // original loaded image (BGR)
    cv::Mat current_image;    // current working image (may be result of erase)
    cv::Mat mask;             // 8-bit single channel mask (255 = erase this)

    // Undo/Redo stacks — each entry stores {image, mask}
    struct Snapshot { cv::Mat image; cv::Mat mask; };
    std::vector<Snapshot> undo_stack;
    std::vector<Snapshot> redo_stack;
    static constexpr int MAX_UNDO = 20;

    void push_undo() {
        if ((int)undo_stack.size() >= MAX_UNDO)
            undo_stack.erase(undo_stack.begin());
        undo_stack.push_back({current_image.clone(), mask.clone()});
        redo_stack.clear();
    }

    // Interaction state
    AppMode    mode      = AppMode::NORMAL;
    DrawTool   tool      = DrawTool::BRUSH;
    double     zoom      = 1.0;
    double     pan_x     = 0.0;
    double     pan_y     = 0.0;
    bool       has_image = false;
    bool       is_drawing = false;
    bool       ctrl_pressed = false;
    bool       is_moving_mask = false;
    bool       drag_started_on_image = false;  // ignore strokes that begin outside the image
    double     drag_start_x = 0.0;
    double     drag_start_y = 0.0;
    double     drag_cur_x   = 0.0;
    double     drag_cur_y   = 0.0;
    cv::Mat    move_mask_snapshot;
    cv::Rect   move_source_snapshot;
    int        brush_radius = 20;
    bool       cursor_on_canvas = false;  // for brush cursor visibility

    // Clone-heal regions (image coords)
    bool       source_rect_set  = false;
    bool       clone_dest_set   = false;
    cv::Rect   source_rect;               // source region in image pixel coords

    // Middle-mouse pan
    bool       is_panning = false;
    double     pan_start_x = 0.0;
    double     pan_start_y = 0.0;
    double     pan_offset_start_x = 0.0;
    double     pan_offset_start_y = 0.0;

    // Current mouse position in canvas widget coords (for zoom-to-cursor)
    double     mouse_x = 0.0;
    double     mouse_y = 0.0;

    // ORT LaMa model (loaded in background thread)
    Ort::Env                            ort_env{ORT_LOGGING_LEVEL_ERROR, "baEraser"};
    Ort::SessionOptions                 ort_opts;
    std::unique_ptr<Ort::Session>       ort_session;
    std::atomic<bool>                   dnn_loaded{false};
    std::atomic<bool>                   dnn_loading{false};
    std::thread                         dnn_thread;
    std::string                         model_path;
    GtkLabel                           *lbl_model_status = nullptr;  // updated from main thread via g_idle_add

    // Current file path
    std::string current_file_path;

    // Cached cairo surfaces — rebuilt only when image/mask changes, never in draw()
    cairo_surface_t *img_surface  = nullptr;  // full image ARGB32
    cairo_surface_t *mask_surface = nullptr;  // mask overlay ARGB32
    cv::Mat          img_surface_bgra;        // keeps pixel data alive for img_surface
    std::vector<uchar> mask_surface_buf;      // keeps pixel data alive for mask_surface

    ~AppState() {
        if (dnn_thread.joinable())   dnn_thread.join();
        if (erase_thread.joinable()) erase_thread.join();
        if (img_surface)  cairo_surface_destroy(img_surface);
        if (mask_surface) cairo_surface_destroy(mask_surface);
    }
};

struct EraseJobResult {
    AppState *app = nullptr;
    cv::Mat result;
    std::string status_message;
};

// ─────────────────────────────────────────────
// Coordinate helpers
// ─────────────────────────────────────────────
// Convert canvas widget coordinates → image pixel coordinates
static cv::Point canvas_to_image(const AppState *app, double cx, double cy) {
    int iw = app->current_image.cols;
    int ih = app->current_image.rows;
    int ww, wh;
    ww = gtk_widget_get_width(GTK_WIDGET(app->canvas));
    wh = gtk_widget_get_height(GTK_WIDGET(app->canvas));

    double img_draw_x = (ww - iw * app->zoom) / 2.0 + app->pan_x;
    double img_draw_y = (wh - ih * app->zoom) / 2.0 + app->pan_y;

    int px = (int)((cx - img_draw_x) / app->zoom);
    int py = (int)((cy - img_draw_y) / app->zoom);
    return {px, py};
}

// ─────────────────────────────────────────────
// Inpainting method implementations
// ─────────────────────────────────────────────

// ── LaMa model loading (async) ────────────────

// Called on main thread via g_idle_add once the background thread finishes
static gboolean lama_load_done_cb(gpointer user_data) {
    AppState *app = static_cast<AppState*>(user_data);
    if (app->dnn_thread.joinable()) app->dnn_thread.join();

    if (app->lbl_model_status) {


        if (app->dnn_loaded) {
            gtk_label_set_text(app->lbl_model_status, "LaMa model loaded ✓");
            gtk_widget_remove_css_class(GTK_WIDGET(app->lbl_model_status), "dim-label");
            gtk_widget_add_css_class(GTK_WIDGET(app->lbl_model_status), "success");
            if (app->method_dropdown &&
                gtk_drop_down_get_selected(app->method_dropdown) == (guint)InpaintMethod::PATCHMATCH) {
                gtk_drop_down_set_selected(app->method_dropdown, (guint)InpaintMethod::DNN_LAMA);
            }
        } else {
            gtk_label_set_text(app->lbl_model_status,
                "LaMa: model not found.\nRun: models/download_lama.sh");
        }


    }
    return G_SOURCE_REMOVE;
}

// Spawns background thread to load LaMa model via ONNX Runtime (no cv::dnn)
static void load_lama_model_async(AppState *app) {
    if (app->dnn_loading || app->dnn_loaded) return;
    app->dnn_loading = true;

    app->dnn_thread = std::thread([app]() {
        std::vector<std::string> search_paths = {
            app->model_path + "/inpainting_lama_2025jan.onnx",
            app->model_path + "/lama_carve_fp32.onnx",
            app->model_path + "/lama_fp32.onnx",
            app->model_path + "/big-lama.onnx",
            "/usr/share/baEraser/models/inpainting_lama_2025jan.onnx",
            "/usr/share/baEraser/models/lama_carve_fp32.onnx",
            "/usr/share/baEraser/models/lama_fp32.onnx",
            std::string(g_get_user_data_dir()) + "/baEraser/models/inpainting_lama_2025jan.onnx",
            std::string(g_get_user_data_dir()) + "/baEraser/models/lama_carve_fp32.onnx",
            std::string(g_get_user_data_dir()) + "/baEraser/models/lama_fp32.onnx",
        };
        for (const auto &p : search_paths) {
            if (std::filesystem::exists(p)) {
                try {
                    app->ort_opts.SetIntraOpNumThreads(
                        static_cast<int>(std::thread::hardware_concurrency()));
                    app->ort_opts.SetGraphOptimizationLevel(
                        GraphOptimizationLevel::ORT_ENABLE_ALL);
                    app->ort_session = std::make_unique<Ort::Session>(
                        app->ort_env, p.c_str(), app->ort_opts);
                    app->dnn_loaded = true;
                    g_print("LaMa model loaded (ORT): %s\n", p.c_str());
                    break;
                } catch (const Ort::Exception &e) {
                    g_print("Failed to load model %s: %s\n", p.c_str(), e.what());
                }
            }
        }
        app->dnn_loading = false;
        // Schedule UI update on main thread
        g_idle_add(lama_load_done_cb, app);
    });
}

// ── Method 2: Multi-scale TELEA ───────────────
// Better than plain TELEA for large masked regions: runs inpainting
// iteratively at decreasing resolutions (coarse-to-fine).
static cv::Mat inpaint_multiscale(const cv::Mat &bgr, const cv::Mat &mask8) {
    constexpr int SCALES = 4;
    constexpr int RADIUS = 5;

    // Build Gaussian pyramid
    std::vector<cv::Mat> img_pyr(SCALES), mask_pyr(SCALES);
    img_pyr[0]  = bgr.clone();
    mask_pyr[0] = mask8.clone();
    for (int s = 1; s < SCALES; s++) {
        cv::pyrDown(img_pyr[s-1],  img_pyr[s]);
        cv::pyrDown(mask_pyr[s-1], mask_pyr[s]);
        cv::threshold(mask_pyr[s], mask_pyr[s], 10, 255, cv::THRESH_BINARY);
    }

    // Inpaint coarsest level first
    cv::Mat current;
    cv::inpaint(img_pyr[SCALES-1], mask_pyr[SCALES-1], current, RADIUS, cv::INPAINT_TELEA);

    // Refine upward
    for (int s = SCALES-2; s >= 0; s--) {
        cv::Mat upsampled;
        cv::pyrUp(current, upsampled, img_pyr[s].size());

        // Blend: use inpainted coarse result as initialisation for finer level
        cv::Mat init = img_pyr[s].clone();
        upsampled.copyTo(init, mask_pyr[s]);

        cv::inpaint(init, mask_pyr[s], current, RADIUS, cv::INPAINT_TELEA);
    }
    return current;
}

// ── Method 3: PatchMatch-style inpainting ─────
// Searches for the best-matching patch from unmasked regions.
// Good for structured backgrounds (grass, sky, brick walls).
static cv::Mat inpaint_patchmatch(const cv::Mat &bgr, const cv::Mat &mask8) {
    constexpr int PATCH   = 7;   // patch radius
    constexpr int HALF    = PATCH / 2;
    // constexpr int ITERS = 5;   // propagation iterations per pass (unused, kept for ref)
    constexpr int PASSES  = 3;   // random search passes

    cv::Mat result = bgr.clone();
    cv::Mat filled = mask8.clone(); // pixels still needing fill = 255

    // Dilate mask slightly to get clean border
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {3,3});

    // Initial fill with TELEA to give something to refine
    cv::inpaint(bgr, mask8, result, 3, cv::INPAINT_TELEA);

    int rows = bgr.rows, cols = bgr.cols;

    // PatchMatch NNF (nearest-neighbour field): for each masked pixel store
    // offset to best source patch
    cv::Mat nnf_dx(rows, cols, CV_32S, cv::Scalar(0));
    cv::Mat nnf_dy(rows, cols, CV_32S, cv::Scalar(0));
    cv::Mat nnf_dist(rows, cols, CV_32F, cv::Scalar(1e9f));

    // Helper: patch SSD between two positions
    auto patch_dist = [&](int r0, int c0, int r1, int c1) -> float {
        float dist = 0.f;
        int count  = 0;
        for (int dy = -HALF; dy <= HALF; dy++) {
            for (int dx = -HALF; dx <= HALF; dx++) {
                int sr0 = r0+dy, sc0 = c0+dx;
                int sr1 = r1+dy, sc1 = c1+dx;
                if (sr0 < 0 || sr0 >= rows || sc0 < 0 || sc0 >= cols) continue;
                if (sr1 < 0 || sr1 >= rows || sc1 < 0 || sc1 >= cols) continue;
                if (filled.at<uchar>(sr1, sc1) > 0) continue; // source must be known
                auto p0 = result.at<cv::Vec3b>(sr0, sc0);
                auto p1 = result.at<cv::Vec3b>(sr1, sc1);
                for (int c = 0; c < 3; c++) {
                    float d = (float)p0[c] - (float)p1[c];
                    dist += d*d;
                }
                count++;
            }
        }
        return count > 0 ? dist / count : 1e9f;
    };

    std::mt19937 rng(42);

    for (int pass = 0; pass < PASSES; pass++) {
        // Forward scan
        for (int r = HALF; r < rows-HALF; r++) {
            for (int c = HALF; c < cols-HALF; c++) {
                if (filled.at<uchar>(r, c) == 0) continue;

                int best_dr = nnf_dx.at<int>(r, c);
                int best_dc = nnf_dy.at<int>(r, c);
                float best  = nnf_dist.at<float>(r, c);

                // Propagation: try neighbour's offset
                for (auto [nr, nc] : std::vector<std::pair<int,int>>{{r-1,c},{r,c-1}}) {
                    if (nr < 0 || nc < 0) continue;
                    int dr = nnf_dx.at<int>(nr, nc);
                    int dc = nnf_dy.at<int>(nr, nc);
                    int sr = r + dr, sc = c + dc;
                    if (sr < HALF || sr >= rows-HALF || sc < HALF || sc >= cols-HALF) continue;
                    float d = patch_dist(r, c, sr, sc);
                    if (d < best) { best = d; best_dr = dr; best_dc = dc; }
                }

                // Random search
                int search_r = std::max(1, std::max(rows, cols));
                while (search_r > 1) {
                    std::uniform_int_distribution<int> dr_dist(-search_r, search_r);
                    int sr = r + dr_dist(rng);
                    int sc = c + dr_dist(rng);
                    sr = std::clamp(sr, HALF, rows-HALF-1);
                    sc = std::clamp(sc, HALF, cols-HALF-1);
                    if (filled.at<uchar>(sr, sc) == 0) {
                        float d = patch_dist(r, c, sr, sc);
                        if (d < best) { best = d; best_dr = sr-r; best_dc = sc-c; }
                    }
                    search_r /= 2;
                }

                nnf_dx.at<int>(r, c)   = best_dr;
                nnf_dy.at<int>(r, c)   = best_dc;
                nnf_dist.at<float>(r, c) = best;
            }
        }

        // Apply NNF: fill each masked pixel with best-match patch centre pixel
        for (int r = HALF; r < rows-HALF; r++) {
            for (int c = HALF; c < cols-HALF; c++) {
                if (filled.at<uchar>(r, c) == 0) continue;
                int sr = r + nnf_dx.at<int>(r, c);
                int sc = c + nnf_dy.at<int>(r, c);
                sr = std::clamp(sr, 0, rows-1);
                sc = std::clamp(sc, 0, cols-1);
                result.at<cv::Vec3b>(r, c) = result.at<cv::Vec3b>(sr, sc);
            }
        }
    }

    // Final TELEA pass to smooth any remaining seams
    cv::Mat seam_mask;
    cv::erode(mask8, seam_mask, kernel);
    cv::bitwise_xor(mask8, seam_mask, seam_mask);
    if (cv::countNonZero(seam_mask) > 0) {
        cv::Mat smoothed;
        cv::inpaint(result, seam_mask, smoothed, 2, cv::INPAINT_TELEA);
        return smoothed;
    }
    return result;
}

// ── Method 5: Fix It — seamless texture fill from surrounding area ────────
// Finds the most uniform patch adjacent to the masked region, copies it over
// the mask, then uses seamlessClone to blend the edges smoothly.
// Best for: uniform backgrounds (sky, grass, walls, fabric).
static cv::Mat inpaint_fixit(const cv::Mat &bgr, const cv::Mat &mask8) {
    // Bounding rect of the mask
    cv::Rect bbox = cv::boundingRect(mask8);
    if (bbox.area() == 0) return bgr.clone();

    int rows = bgr.rows, cols = bgr.cols;
    int bw = bbox.width, bh = bbox.height;

    // Candidate source regions: 4 directions around the bounding box,
    // same size as bbox, offset by bbox dimension + a small gap.
    constexpr int GAP = 4;
    std::vector<cv::Rect> candidates = {
        { bbox.x,           bbox.y - bh - GAP, bw, bh },  // above
        { bbox.x,           bbox.y + bh + GAP, bw, bh },  // below
        { bbox.x - bw - GAP, bbox.y,           bw, bh },  // left
        { bbox.x + bw + GAP, bbox.y,           bw, bh },  // right
    };

    // Score each candidate by variance (lower = more uniform = better source)
    double best_score = std::numeric_limits<double>::max();
    cv::Rect best_rect;
    for (auto &r : candidates) {
        // Clamp to image bounds
        cv::Rect clamped(
            std::clamp(r.x, 0, cols - 1),
            std::clamp(r.y, 0, rows - 1),
            0, 0);
        clamped.width  = std::clamp(r.x + r.width,  0, cols) - clamped.x;
        clamped.height = std::clamp(r.y + r.height, 0, rows) - clamped.y;
        if (clamped.width < 4 || clamped.height < 4) continue;

        cv::Mat patch = bgr(clamped);
        cv::Mat grey;
        cv::cvtColor(patch, grey, cv::COLOR_BGR2GRAY);
        cv::Scalar mean, stddev;
        cv::meanStdDev(grey, mean, stddev);
        double score = stddev[0];  // lower stddev = more uniform
        if (score < best_score) { best_score = score; best_rect = clamped; }
    }

    // Fallback: if no candidate is valid, use TELEA
    if (best_rect.area() == 0)
        return inpaint_patchmatch(bgr, mask8);

    // Build source patch resized to exactly bbox size
    cv::Mat src_patch;
    cv::resize(bgr(best_rect), src_patch, cv::Size(bw, bh));

    // Compose source image: full copy of bgr with the patch pasted over bbox
    cv::Mat src = bgr.clone();
    src_patch.copyTo(src(bbox));

    // Build a white mask the size of bbox for seamlessClone
    cv::Mat clone_mask(bh, bw, CV_8UC1, cv::Scalar(255));

    // Center point of the destination region
    cv::Point center(bbox.x + bw / 2, bbox.y + bh / 2);

    cv::Mat result;
    try {
        cv::seamlessClone(src_patch, bgr, clone_mask, center,
                          result, cv::MIXED_CLONE);
    } catch (...) {
        // seamlessClone can fail on very small regions — fall back
        result = inpaint_patchmatch(bgr, mask8);
    }
    return result;
}

// ── Method 6: Clone Heal — user-selected source region ───────────────────
// Takes a user-defined source rect (image coords) and seamlessly clones it
// into the masked destination area using cv::seamlessClone (MIXED_CLONE).
// The source patch is resized to match the destination bounding box so the
// blend is always pixel-perfect regardless of size difference.
static cv::Mat inpaint_clone_heal(const cv::Mat &bgr,
                                  const cv::Mat &mask8,
                                  const cv::Rect &src_rect) {
    cv::Rect bbox = cv::boundingRect(mask8);
    if (bbox.area() == 0 || src_rect.area() == 0) return bgr.clone();

    int rows = bgr.rows, cols = bgr.cols;

    // Clamp source rect to image bounds
    cv::Rect src = src_rect & cv::Rect(0, 0, cols, rows);
    if (src.area() == 0) return bgr.clone();

    // Extract and resize source patch to destination bbox size
    cv::Mat src_patch;
    cv::resize(bgr(src), src_patch, cv::Size(bbox.width, bbox.height));

    // White mask covering the full source patch
    cv::Mat clone_mask(bbox.height, bbox.width, CV_8UC1, cv::Scalar(255));

    // Destination center
    cv::Point center(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);

    // seamlessClone needs dest center to be at least patch_size/2 away from edge
    int half_w = bbox.width  / 2;
    int half_h = bbox.height / 2;
    center.x = std::clamp(center.x, half_w + 1, cols - half_w - 1);
    center.y = std::clamp(center.y, half_h + 1, rows - half_h - 1);

    cv::Mat result;
    try {
        cv::seamlessClone(src_patch, bgr, clone_mask, center,
                          result, cv::MIXED_CLONE);
    } catch (...) {
        // Fallback to normal clone without blending
        result = bgr.clone();
        src_patch.copyTo(result(bbox));
    }
    return result;
}

// ── Method 4: LaMa via ONNX Runtime ──────────────
static cv::Mat inpaint_lama(AppState *app, const cv::Mat &bgr, const cv::Mat &mask8,
                            std::string *status_message) {
    if (!app->dnn_loaded || !app->ort_session) {
        g_print("LaMa model not loaded, falling back to PatchMatch\n");
        if (status_message)
            *status_message = "LaMa model was not loaded. PatchMatch fallback applied.";
        return inpaint_patchmatch(bgr, mask8);
    }

    // ── Prepare float32 image blob [1, 3, 512, 512] ──────────────────────────
    cv::Mat img_resized, mask_resized;
    cv::resize(bgr,   img_resized,  {512, 512});
    cv::resize(mask8, mask_resized, {512, 512});

    // LaMa was trained on RGB — convert BGR → RGB before normalising
    cv::Mat img_rgb;
    cv::cvtColor(img_resized, img_rgb, cv::COLOR_BGR2RGB);

    // Normalise image to [0, 1] and convert HWC-RGB → CHW-RGB
    cv::Mat img_f32;
    img_rgb.convertTo(img_f32, CV_32F, 1.0 / 255.0);

    // Split channels and lay them out contiguously: [R plane][G plane][B plane]
    std::vector<cv::Mat> img_ch(3);
    cv::split(img_f32, img_ch);
    std::vector<float> img_data(1 * 3 * 512 * 512);
    for (int c = 0; c < 3; c++) {
        std::memcpy(img_data.data() + c * 512 * 512,
                    img_ch[c].ptr<float>(),
                    512 * 512 * sizeof(float));
    }

    // ── Prepare float32 mask blob [1, 1, 512, 512] ───────────────────────────
    cv::Mat mask_f32;
    cv::threshold(mask_resized, mask_f32, 0, 1.0, cv::THRESH_BINARY);
    mask_f32.convertTo(mask_f32, CV_32F);
    std::vector<float> mask_data(1 * 1 * 512 * 512);
    std::memcpy(mask_data.data(), mask_f32.ptr<float>(), 512 * 512 * sizeof(float));

    // ── Build ORT tensors ─────────────────────────────────────────────────────
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    std::array<int64_t, 4> img_shape  = {1, 3, 512, 512};
    std::array<int64_t, 4> mask_shape = {1, 1, 512, 512};

    Ort::Value img_tensor = Ort::Value::CreateTensor<float>(
        mem_info, img_data.data(), img_data.size(),
        img_shape.data(), img_shape.size());
    Ort::Value mask_tensor = Ort::Value::CreateTensor<float>(
        mem_info, mask_data.data(), mask_data.size(),
        mask_shape.data(), mask_shape.size());

    // ── Run inference ─────────────────────────────────────────────────────────
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(img_tensor));
    inputs.push_back(std::move(mask_tensor));

    // Query actual input/output names from the model
    Ort::AllocatorWithDefaultOptions allocator;
    std::string in0_name = app->ort_session->GetInputNameAllocated(0, allocator).get();
    std::string in1_name = app->ort_session->GetInputNameAllocated(1, allocator).get();
    std::string out0_name = app->ort_session->GetOutputNameAllocated(0, allocator).get();

    const char *input_names[]  = {in0_name.c_str(), in1_name.c_str()};
    const char *output_names[] = {out0_name.c_str()};

    std::vector<Ort::Value> outputs;
    try {
        outputs = app->ort_session->Run(
            Ort::RunOptions{nullptr},
            input_names,  inputs.data(),  2,
            output_names, 1);
    } catch (const Ort::Exception &e) {
        g_print("LaMa ORT forward failed: %s – falling back to PatchMatch\n", e.what());
        if (status_message)
            *status_message = "LaMa failed. PatchMatch fallback applied.";
        return inpaint_patchmatch(bgr, mask8);
    }

    // ── Decode output [1, 3, 512, 512] → BGR uint8 ───────────────────────────
    const float *out_ptr = outputs[0].GetTensorData<float>();
    auto out_info = outputs[0].GetTensorTypeAndShapeInfo();
    size_t out_count = out_info.GetElementCount();
    if (out_count < 3 * 512 * 512) {
        g_printerr("LaMa output tensor is too small, falling back to PatchMatch\n");
        if (status_message)
            *status_message = "LaMa returned an invalid output. PatchMatch fallback applied.";
        return inpaint_patchmatch(bgr, mask8);
    }

    double out_min = out_ptr[0];
    double out_max = out_ptr[0];
    for (size_t i = 1; i < out_count; i++) {
        out_min = std::min(out_min, (double)out_ptr[i]);
        out_max = std::max(out_max, (double)out_ptr[i]);
    }
    double output_scale = out_max <= 1.5 ? 255.0 : 1.0;

    // out_ptr layout: [R plane][G plane][B plane], each 512*512 floats.
    // Most bundled LaMa exports return [0,255], but some exports return [0,1].
    std::vector<cv::Mat> out_ch(3);
    for (int c = 0; c < 3; c++) {
        cv::Mat plane(512, 512, CV_32F,
                      const_cast<float*>(out_ptr + c * 512 * 512));
        plane.convertTo(out_ch[c], CV_8U, output_scale);
    }
    // out_ch[0]=R, out_ch[1]=G, out_ch[2]=B — merge then convert RGB→BGR
    cv::Mat merged_rgb;
    cv::merge(out_ch, merged_rgb);  // uint8 RGB 512x512
    cv::Mat merged;
    cv::cvtColor(merged_rgb, merged, cv::COLOR_RGB2BGR);  // → BGR

    // Resize back to original image dimensions
    cv::Mat result_full;
    cv::resize(merged, result_full, {bgr.cols, bgr.rows});

    // Composite: only replace masked pixels
    cv::Mat final_result = bgr.clone();
    cv::Mat mask_bool;
    cv::threshold(mask8, mask_bool, 0, 255, cv::THRESH_BINARY);
    result_full.copyTo(final_result, mask_bool);

    cv::Mat diff;
    cv::absdiff(bgr, final_result, diff);
    cv::Scalar diff_mean = cv::mean(diff, mask_bool);
    double masked_delta = (diff_mean[0] + diff_mean[1] + diff_mean[2]) / 3.0;
    if (masked_delta < 1.0) {
        g_printerr("LaMa returned an unchanged result, falling back to PatchMatch\n");
        if (status_message)
            *status_message = "LaMa returned an unchanged result. PatchMatch fallback applied.";
        return inpaint_patchmatch(bgr, mask8);
    }

    if (status_message)
        *status_message = "Done. LaMa AI applied.";
    return final_result;
}

// ─────────────────────────────────────────────
// GdkPixbuf ↔ cv::Mat conversions
// ─────────────────────────────────────────────
static cv::Mat gdk_pixbuf_to_mat(GdkPixbuf *pixbuf) {
    int width    = gdk_pixbuf_get_width(pixbuf);
    int height   = gdk_pixbuf_get_height(pixbuf);
    int channels = gdk_pixbuf_get_n_channels(pixbuf);
    int rowstride = gdk_pixbuf_get_rowstride(pixbuf);
    guchar *pixels = gdk_pixbuf_get_pixels(pixbuf);

    cv::Mat mat;
    if (channels == 3) {
        cv::Mat tmp(height, width, CV_8UC3, pixels, rowstride);
        cv::cvtColor(tmp, mat, cv::COLOR_RGB2BGR);
    } else if (channels == 4) {
        cv::Mat tmp(height, width, CV_8UC4, pixels, rowstride);
        cv::cvtColor(tmp, mat, cv::COLOR_RGBA2BGR);
    } else {
        // Grayscale → BGR
        cv::Mat tmp(height, width, CV_8UC1, pixels, rowstride);
        cv::cvtColor(tmp, mat, cv::COLOR_GRAY2BGR);
    }
    return mat.clone();
}



// ─────────────────────────────────────────────
// Image loading
// ─────────────────────────────────────────────
static void load_image_from_path(AppState *app, const std::string &path) {
    GError *error = nullptr;
    GdkPixbuf *pixbuf = gdk_pixbuf_new_from_file(path.c_str(), &error);
    if (!pixbuf || error) {
        std::string msg = error ? error->message : "Unknown error";
        if (error) g_error_free(error);
        show_error_dialog(app, "Cannot open image:\n" + msg);
        return;
    }

    app->current_file_path = path;
    app->original_image = gdk_pixbuf_to_mat(pixbuf);
    app->current_image  = app->original_image.clone();
    app->undo_stack.clear();
    app->redo_stack.clear();
    app->has_image = true;
    app->mode = AppMode::NORMAL;
    app->source_rect_set = false;
    clear_mask(app);
    app->zoom  = 1.0;
    app->pan_x = 0.0;
    app->pan_y = 0.0;

    // Hide system cursor — we draw our own brush ring
    gtk_widget_set_cursor_from_name(GTK_WIDGET(app->canvas), "none");

    g_object_unref(pixbuf);
    rebuild_image_surface(app);
    update_button_states(app);

    // Status
    std::string filename = std::filesystem::path(path).filename().string();
    std::string status = filename + "  " +
        std::to_string(app->current_image.cols) + "×" +
        std::to_string(app->current_image.rows) + " px";
    gtk_label_set_text(app->status_label, status.c_str());

    // Schedule fit-to-window after layout pass
    g_idle_add(fit_image_idle, app);
}

static void update_canvas_size(AppState *app) {
    if (!app->has_image) return;
    // The image position is controlled only by pan_x/pan_y. Keep the drawing
    // area as a stable viewport so GtkScrolledWindow cannot scroll the image
    // independently while the user is drawing a mask.
    gtk_widget_set_size_request(GTK_WIDGET(app->canvas), 1, 1);
}

// Fit image into the scrolled-window viewport (called via g_idle_add after layout)
static gboolean fit_image_idle(gpointer user_data) {
    AppState *app = static_cast<AppState*>(user_data);
    if (!app->has_image) return G_SOURCE_REMOVE;

    int vw = gtk_widget_get_width(app->scrolled_window);
    int vh = gtk_widget_get_height(app->scrolled_window);
    if (vw < 2 || vh < 2) return G_SOURCE_REMOVE; // layout not ready yet, retry

    double zx = (double)vw / app->current_image.cols;
    double zy = (double)vh / app->current_image.rows;
    app->zoom  = std::min(zx, zy) * 0.97;
    app->pan_x = 0.0;
    app->pan_y = 0.0;

    char buf[32];
    snprintf(buf, sizeof(buf), "%.0f%%", app->zoom * 100.0);
    gtk_label_set_text(app->zoom_label, buf);

    update_canvas_size(app);
    gtk_widget_queue_draw(GTK_WIDGET(app->canvas));
    return G_SOURCE_REMOVE;
}

// ─────────────────────────────────────────────
// Button state management
// ─────────────────────────────────────────────
static void update_button_states(AppState *app) {
    bool clone_active = app->source_rect_set || app->clone_dest_set ||
        app->mode == AppMode::PICK_SOURCE || app->mode == AppMode::PICK_DEST;
    gtk_widget_set_sensitive(GTK_WIDGET(app->btn_save),        app->has_image);
    gtk_widget_set_sensitive(GTK_WIDGET(app->btn_erase),       app->has_image && !clone_active);
    gtk_widget_set_sensitive(GTK_WIDGET(app->btn_fixit),       app->has_image && !clone_active);
    gtk_widget_set_sensitive(GTK_WIDGET(app->btn_clear_mask),  app->has_image);
    gtk_widget_set_sensitive(GTK_WIDGET(app->btn_pick_source),   app->has_image);
    gtk_widget_set_sensitive(GTK_WIDGET(app->btn_clone_here),    app->has_image && app->source_rect_set && app->clone_dest_set);
    gtk_widget_set_sensitive(GTK_WIDGET(app->btn_cancel_clone),  app->has_image &&
        (app->source_rect_set || app->clone_dest_set ||
         app->mode == AppMode::PICK_SOURCE || app->mode == AppMode::PICK_DEST));
    gtk_widget_set_sensitive(GTK_WIDGET(app->btn_undo),        !app->undo_stack.empty());
    gtk_widget_set_sensitive(GTK_WIDGET(app->btn_redo),        !app->redo_stack.empty());
    update_clone_ui(app);
}

static void update_clone_ui(AppState *app) {
    if (!app->lbl_clone_source || !app->lbl_clone_dest || !app->lbl_clone_hint)
        return;

    if (!app->has_image) {
        gtk_label_set_text(app->lbl_clone_source, "1. Source: open an image first");
        gtk_label_set_text(app->lbl_clone_dest,   "2. Destination: waiting");
        gtk_label_set_text(app->lbl_clone_hint,   "Clone Heal copies texture from one rectangle into another.");
        return;
    }

    if (app->mode == AppMode::PICK_SOURCE) {
        gtk_label_set_text(app->lbl_clone_source, "1. Source: draw a green rectangle now");
        gtk_label_set_text(app->lbl_clone_dest,   "2. Destination: next step");
        gtk_label_set_text(app->lbl_clone_hint,   "Drag over the clean area you want to copy from.");
    } else if (app->mode == AppMode::PICK_DEST) {
        gtk_label_set_text(app->lbl_clone_source, "1. Source: selected");
        gtk_label_set_text(app->lbl_clone_dest,
            app->clone_dest_set ? "2. Destination: selected" : "2. Destination: draw a blue rectangle now");
        gtk_label_set_text(app->lbl_clone_hint,
            app->clone_dest_set ? "Ready. Click Clone Here, or draw destination again to adjust."
                                : "Drag over the place where the source should be blended.");
    } else {
        gtk_label_set_text(app->lbl_clone_source,
            app->source_rect_set ? "1. Source: selected" : "1. Source: not selected");
        gtk_label_set_text(app->lbl_clone_dest,
            app->clone_dest_set ? "2. Destination: selected" : "2. Destination: not selected");
        gtk_label_set_text(app->lbl_clone_hint,
            (app->source_rect_set && app->clone_dest_set)
                ? "Ready. Click Clone Here to apply."
                : "Click Pick Source to start a guided clone-heal.");
    }
}

// ─────────────────────────────────────────────
// Cached surface builders
// ─────────────────────────────────────────────

// Call whenever current_image changes (load, erase, undo/redo)
static void rebuild_image_surface(AppState *app) {
    if (app->img_surface) {
        cairo_surface_destroy(app->img_surface);
        app->img_surface = nullptr;
    }
    if (!app->has_image || app->current_image.empty()) return;

    // BGR → BGRA (alpha=255). Keep mat alive in app->img_surface_bgra.
    cv::cvtColor(app->current_image, app->img_surface_bgra, cv::COLOR_BGR2BGRA);

    int cols   = app->img_surface_bgra.cols;
    int rows   = app->img_surface_bgra.rows;
    int stride = cairo_format_stride_for_width(CAIRO_FORMAT_ARGB32, cols);

    // If OpenCV row-stride differs from Cairo's required stride, copy into
    // a tightly-packed Cairo-stride buffer stored in img_surface_bgra itself.
    if (stride != (int)app->img_surface_bgra.step) {
        cv::Mat aligned(rows, cols, CV_8UC4);
        for (int r = 0; r < rows; r++)
            memcpy(aligned.ptr(r), app->img_surface_bgra.ptr(r), (size_t)cols * 4);
        app->img_surface_bgra = aligned;
        // recalc — now step == cols*4, check if that matches
        stride = cairo_format_stride_for_width(CAIRO_FORMAT_ARGB32, cols);
    }

    app->img_surface = cairo_image_surface_create_for_data(
        app->img_surface_bgra.data, CAIRO_FORMAT_ARGB32,
        cols, rows, (int)app->img_surface_bgra.step);
}

// Call whenever mask changes (brush stroke, rect, clear)
static void rebuild_mask_surface(AppState *app) {
    if (app->mask_surface) {
        cairo_surface_destroy(app->mask_surface);
        app->mask_surface = nullptr;
    }
    if (app->mask.empty() || cv::countNonZero(app->mask) == 0) return;

    int cols    = app->mask.cols;
    int rows    = app->mask.rows;
    int mstride = cairo_format_stride_for_width(CAIRO_FORMAT_ARGB32, cols);

    app->mask_surface_buf.assign((size_t)mstride * rows, 0);
    uchar *buf = app->mask_surface_buf.data();

    for (int y = 0; y < rows; y++) {
        const uchar *mp  = app->mask.ptr(y);
        uchar       *out = buf + y * mstride;
        for (int x = 0; x < cols; x++) {
            if (mp[x] > 0) {
                // Pre-multiplied BGRA: red at ~45% opacity
                // A=115, pre_R=115, pre_G=0, pre_B=0
                out[x*4+0] = 0;
                out[x*4+1] = 0;
                out[x*4+2] = 115;
                out[x*4+3] = 115;
            }
        }
    }

    app->mask_surface = cairo_image_surface_create_for_data(
        buf, CAIRO_FORMAT_ARGB32, cols, rows, mstride);
}

static void finish_mask_change(AppState *app) {
    rebuild_mask_surface(app);
    if (app->canvas)
        gtk_widget_queue_draw(GTK_WIDGET(app->canvas));
}

static void cancel_mask_move(AppState *app) {
    app->is_moving_mask = false;
    app->move_mask_snapshot.release();
}

static void set_mask(AppState *app, cv::Mat mask, bool clear_clone_destination) {
    app->mask = std::move(mask);
    if (clear_clone_destination)
        app->clone_dest_set = false;
    cancel_mask_move(app);
    finish_mask_change(app);
}

static void clear_mask(AppState *app, bool clear_clone_destination) {
    if (!app->has_image || app->current_image.empty()) return;
    set_mask(app,
        cv::Mat::zeros(app->current_image.rows, app->current_image.cols, CV_8UC1),
        clear_clone_destination);
}

// ─────────────────────────────────────────────
// Canvas drawing
// ─────────────────────────────────────────────
static void canvas_draw(GtkDrawingArea *da, cairo_t *cr, int w, int h, gpointer user_data) {
    AppState *app = static_cast<AppState*>(user_data);

    // Background
    cairo_set_source_rgb(cr, 0.15, 0.15, 0.15);
    cairo_paint(cr);

    if (!app->has_image) {
        // Draw drop hint
        cairo_set_source_rgba(cr, 1, 1, 1, 0.3);
        cairo_select_font_face(cr, "Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_NORMAL);
        cairo_set_font_size(cr, 18);
        const char *hint = "Drop an image here or click Open";
        cairo_text_extents_t ext;
        cairo_text_extents(cr, hint, &ext);
        cairo_move_to(cr, (w - ext.width) / 2.0, h / 2.0);
        cairo_show_text(cr, hint);
        return;
    }

    // Compute draw position (centered + pan)
    int iw = app->current_image.cols;
    int ih = app->current_image.rows;
    double draw_x = (w - iw * app->zoom) / 2.0 + app->pan_x;
    double draw_y = (h - ih * app->zoom) / 2.0 + app->pan_y;

    cairo_save(cr);
    cairo_translate(cr, draw_x, draw_y);
    cairo_scale(cr, app->zoom, app->zoom);

    // Draw image from cached surface (no allocation in draw path)
    if (app->img_surface) {
        cairo_set_source_surface(cr, app->img_surface, 0, 0);
        cairo_paint(cr);
    }

    // Draw mask overlay from cached surface
    if (app->mask_surface) {
        cairo_set_source_surface(cr, app->mask_surface, 0, 0);
        cairo_paint(cr);
    }

    cairo_restore(cr);

    // Draw rect selection preview (widget coords)
    if (app->tool == DrawTool::RECT && app->is_drawing &&
        !app->is_moving_mask && !app->is_panning) {
        cairo_save(cr);
        double wx0 = draw_x + app->drag_start_x * app->zoom;
        double wy0 = draw_y + app->drag_start_y * app->zoom;
        double wx1 = draw_x + app->drag_cur_x   * app->zoom;
        double wy1 = draw_y + app->drag_cur_y   * app->zoom;
        // PICK_SOURCE mode: green; PICK_DEST/normal rect mode: orange
        if (app->mode == AppMode::PICK_SOURCE)
            cairo_set_source_rgba(cr, 0.2, 1.0, 0.2, 0.9);
        else if (app->mode == AppMode::PICK_DEST)
            cairo_set_source_rgba(cr, 0.1, 0.55, 1.0, 0.9);
        else
            cairo_set_source_rgba(cr, 1.0, 0.5, 0.0, 0.85);
        cairo_set_line_width(cr, 1.5);
        const double dashes[] = {6, 3};
        cairo_set_dash(cr, dashes, 2, 0);
        cairo_rectangle(cr,
            std::min(wx0, wx1), std::min(wy0, wy1),
            std::abs(wx1 - wx0), std::abs(wy1 - wy0));
        cairo_stroke(cr);
        cairo_restore(cr);
    }

    // Draw confirmed source rect (green solid, always visible when set)
    if (app->source_rect_set && !app->current_image.empty()) {
        cairo_save(cr);
        cv::Rect &sr = app->source_rect;
        double sx = draw_x + sr.x * app->zoom;
        double sy = draw_y + sr.y * app->zoom;
        double sw = sr.width  * app->zoom;
        double sh = sr.height * app->zoom;
        // Semi-transparent green fill
        cairo_set_source_rgba(cr, 0.2, 0.9, 0.2, 0.15);
        cairo_rectangle(cr, sx, sy, sw, sh);
        cairo_fill(cr);
        // Green dashed border
        cairo_set_source_rgba(cr, 0.2, 0.9, 0.2, 0.9);
        cairo_set_line_width(cr, 1.5);
        const double dashes2[] = {6, 3};
        cairo_set_dash(cr, dashes2, 2, 0);
        cairo_rectangle(cr, sx, sy, sw, sh);
        cairo_stroke(cr);
        // "SOURCE" label
        cairo_set_dash(cr, nullptr, 0, 0);
        cairo_set_source_rgba(cr, 0.2, 0.9, 0.2, 0.95);
        cairo_set_font_size(cr, 11);
        cairo_move_to(cr, sx + 4, sy + 14);
        cairo_show_text(cr, "SOURCE");
        cairo_restore(cr);
    }

    // Draw confirmed clone destination (blue outline) so it is distinct from erase masks
    if (app->clone_dest_set && !app->mask.empty()) {
        cv::Rect dr = cv::boundingRect(app->mask);
        if (dr.area() > 0) {
            cairo_save(cr);
            double dx = draw_x + dr.x * app->zoom;
            double dy = draw_y + dr.y * app->zoom;
            double dw = dr.width  * app->zoom;
            double dh = dr.height * app->zoom;
            cairo_set_source_rgba(cr, 0.1, 0.55, 1.0, 0.16);
            cairo_rectangle(cr, dx, dy, dw, dh);
            cairo_fill(cr);
            cairo_set_source_rgba(cr, 0.1, 0.55, 1.0, 0.95);
            cairo_set_line_width(cr, 1.8);
            const double dashes3[] = {7, 4};
            cairo_set_dash(cr, dashes3, 2, 0);
            cairo_rectangle(cr, dx, dy, dw, dh);
            cairo_stroke(cr);
            cairo_set_dash(cr, nullptr, 0, 0);
            cairo_set_font_size(cr, 11);
            cairo_move_to(cr, dx + 4, dy + 14);
            cairo_show_text(cr, "DESTINATION");
            cairo_restore(cr);
        }
    }

    // Draw custom cursor — visible when cursor is on canvas (or frozen at last pos)
    if (app->cursor_on_canvas && !app->ctrl_pressed &&
        !app->is_moving_mask && !app->is_panning) {
        cairo_save(cr);

        if (app->tool == DrawTool::BRUSH) {
            double r = app->brush_radius * app->zoom;

            // Outer white ring
            cairo_arc(cr, app->mouse_x, app->mouse_y, r, 0, 2 * G_PI);
            cairo_set_source_rgba(cr, 1.0, 1.0, 1.0, 0.9);
            cairo_set_line_width(cr, 2.0);
            cairo_set_dash(cr, nullptr, 0, 0);
            cairo_stroke(cr);

            // Inner dark ring
            cairo_arc(cr, app->mouse_x, app->mouse_y, r, 0, 2 * G_PI);
            cairo_set_source_rgba(cr, 0.0, 0.0, 0.0, 0.6);
            cairo_set_line_width(cr, 1.0);
            cairo_stroke(cr);

            // Centre dot
            cairo_arc(cr, app->mouse_x, app->mouse_y, 1.5, 0, 2 * G_PI);
            cairo_set_source_rgba(cr, 1.0, 1.0, 1.0, 0.9);
            cairo_fill(cr);

        } else {
            // RECT tool — crosshair cursor
            constexpr double ARM = 10.0;
            double cx = app->mouse_x;
            double cy = app->mouse_y;

            cairo_set_line_width(cr, 2.0);
            cairo_set_dash(cr, nullptr, 0, 0);

            // White outline for contrast
            cairo_set_source_rgba(cr, 1.0, 1.0, 1.0, 0.9);
            cairo_move_to(cr, cx - ARM, cy); cairo_line_to(cr, cx + ARM, cy);
            cairo_move_to(cr, cx, cy - ARM); cairo_line_to(cr, cx, cy + ARM);
            cairo_stroke(cr);

            // Dark inner line
            cairo_set_source_rgba(cr, 0.0, 0.0, 0.0, 0.8);
            cairo_set_line_width(cr, 1.0);
            cairo_move_to(cr, cx - ARM, cy); cairo_line_to(cr, cx + ARM, cy);
            cairo_move_to(cr, cx, cy - ARM); cairo_line_to(cr, cx, cy + ARM);
            cairo_stroke(cr);
        }

        cairo_restore(cr);
    }
}

// ─────────────────────────────────────────────
// Mask drawing helpers
// ─────────────────────────────────────────────
static void draw_brush_on_mask(AppState *app, int px, int py) {
    if (!app->has_image) return;
    int r = app->brush_radius;
    cv::circle(app->mask, {px, py}, r, cv::Scalar(255), -1, cv::LINE_AA);
    finish_mask_change(app);
}

static void draw_rect_on_mask(AppState *app, cv::Point p0, cv::Point p1) {
    if (!app->has_image) return;
    int x0 = std::clamp(std::min(p0.x, p1.x), 0, app->mask.cols - 1);
    int y0 = std::clamp(std::min(p0.y, p1.y), 0, app->mask.rows - 1);
    int x1 = std::clamp(std::max(p0.x, p1.x), 0, app->mask.cols - 1);
    int y1 = std::clamp(std::max(p0.y, p1.y), 0, app->mask.rows - 1);
    cv::rectangle(app->mask, {x0, y0}, {x1, y1}, cv::Scalar(255), -1);
    finish_mask_change(app);
}

static cv::Mat translate_mask(const cv::Mat &mask, int dx, int dy) {
    cv::Mat moved = cv::Mat::zeros(mask.rows, mask.cols, mask.type());
    cv::Mat transform = (cv::Mat_<double>(2, 3) << 1, 0, dx, 0, 1, dy);
    cv::warpAffine(mask, moved, transform, mask.size(),
                   cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
    return moved;
}

static bool has_move_target(const AppState *app) {
    return app->has_image && !app->mask.empty() && cv::countNonZero(app->mask) > 0;
}

static bool image_point_in_bounds(const AppState *app, const cv::Point &p) {
    return app->has_image &&
           p.x >= 0 && p.y >= 0 &&
           p.x < app->current_image.cols &&
           p.y < app->current_image.rows;
}

static bool mask_contains_point(const AppState *app, const cv::Point &p) {
    return image_point_in_bounds(app, p) &&
           has_move_target(app) &&
           app->mask.at<uchar>(p.y, p.x) > 0;
}

static bool is_control_key(guint keyval) {
    return keyval == GDK_KEY_Control_L || keyval == GDK_KEY_Control_R;
}

static bool event_has_ctrl(GtkEventController *controller) {
    GdkModifierType state = gtk_event_controller_get_current_event_state(controller);
    return (state & GDK_CONTROL_MASK) != 0;
}

static void update_canvas_cursor(AppState *app) {
    if (!app->has_image) return;
    const char *cursor = "none";
    if (app->is_moving_mask || app->is_panning) {
        cursor = "grabbing";
    } else if (app->ctrl_pressed) {
        cursor = "grab";
    }
    gtk_widget_set_cursor_from_name(GTK_WIDGET(app->canvas), cursor);
    gtk_widget_queue_draw(GTK_WIDGET(app->canvas));
}

static void begin_mask_move(AppState *app, const cv::Point &ip) {
    app->push_undo();
    app->is_moving_mask = true;
    app->is_drawing = true;
    app->drag_started_on_image = true;
    app->drag_start_x = ip.x;
    app->drag_start_y = ip.y;
    app->drag_cur_x   = ip.x;
    app->drag_cur_y   = ip.y;
    app->move_mask_snapshot = app->mask.clone();
    app->move_source_snapshot = app->source_rect;
    gtk_label_set_text(app->status_label, "Moving selection…");
    update_canvas_cursor(app);
}

static void begin_pan(AppState *app, double x, double y) {
    app->is_panning = true;
    app->is_drawing = true;
    app->drag_started_on_image = true;
    app->pan_start_x = x;
    app->pan_start_y = y;
    app->pan_offset_start_x = app->pan_x;
    app->pan_offset_start_y = app->pan_y;
    update_canvas_cursor(app);
}

static void begin_move_or_pan(AppState *app, double x, double y) {
    cv::Point ip = canvas_to_image(app, x, y);
    if (mask_contains_point(app, ip)) {
        begin_mask_move(app, ip);
    } else {
        begin_pan(app, x, y);
    }
}

// ─────────────────────────────────────────────
// Gesture / input callbacks
// ─────────────────────────────────────────────

// ── Left-button drag: draw brush / rect ──────
static void on_draw_drag_begin(GtkGestureDrag *gesture, double x, double y, gpointer user_data) {
    AppState *app = static_cast<AppState*>(user_data);
    app->drag_started_on_image = false;
    if (!app->has_image) return;
    gtk_widget_grab_focus(GTK_WIDGET(app->canvas));
    app->ctrl_pressed = app->ctrl_pressed || event_has_ctrl(GTK_EVENT_CONTROLLER(gesture));
    update_canvas_cursor(app);

    if (app->ctrl_pressed) {
        begin_move_or_pan(app, x, y);
        return;
    }

    cv::Point ip = canvas_to_image(app, x, y);

    // Ignore drags that start outside the image bounds
    if (!image_point_in_bounds(app, ip)) return;

    // Only now commit to this stroke
    app->drag_started_on_image = true;

    if (app->mode == AppMode::NORMAL)
        app->push_undo();
    app->is_drawing = true;
    app->drag_start_x = ip.x;
    app->drag_start_y = ip.y;
    app->drag_cur_x   = ip.x;
    app->drag_cur_y   = ip.y;

    if (app->tool == DrawTool::BRUSH && app->mode == AppMode::NORMAL) {
        draw_brush_on_mask(app, ip.x, ip.y);
        gtk_widget_queue_draw(GTK_WIDGET(app->canvas));
    }
}

static void on_draw_drag_update(GtkGestureDrag *gesture, double dx, double dy, gpointer user_data) {
    AppState *app = static_cast<AppState*>(user_data);
    if (!app->has_image || !app->is_drawing || !app->drag_started_on_image) return;

    double start_x, start_y;
    gtk_gesture_drag_get_start_point(gesture, &start_x, &start_y);

    double cx = start_x + dx;
    double cy = start_y + dy;
    cv::Point ip = canvas_to_image(app, cx, cy);
    app->drag_cur_x = ip.x;
    app->drag_cur_y = ip.y;

    if (app->is_moving_mask) {
        int mdx = (int)std::round(app->drag_cur_x - app->drag_start_x);
        int mdy = (int)std::round(app->drag_cur_y - app->drag_start_y);
        app->mask = translate_mask(app->move_mask_snapshot, mdx, mdy);
        if (app->source_rect_set &&
            app->mode != AppMode::PICK_DEST &&
            !app->move_source_snapshot.empty()) {
            cv::Rect img_bounds(0, 0, app->current_image.cols, app->current_image.rows);
            app->source_rect = (app->move_source_snapshot + cv::Point(mdx, mdy)) & img_bounds;
        }
        finish_mask_change(app);
        return;
    }

    if (app->is_panning) {
        app->pan_x = app->pan_offset_start_x + dx;
        app->pan_y = app->pan_offset_start_y + dy;
        gtk_widget_queue_draw(GTK_WIDGET(app->canvas));
        return;
    }

    if (app->tool == DrawTool::BRUSH && app->mode == AppMode::NORMAL) {
        draw_brush_on_mask(app, ip.x, ip.y);
    }
    gtk_widget_queue_draw(GTK_WIDGET(app->canvas));
}

static void on_draw_drag_end(GtkGestureDrag *gesture, double dx, double dy, gpointer user_data) {
    AppState *app = static_cast<AppState*>(user_data);
    (void)gesture; (void)dx; (void)dy;
    if (!app->is_drawing || !app->drag_started_on_image) {
        app->is_drawing = false;
        return;
    }
    app->is_drawing = false;

    if (app->is_moving_mask) {
        cancel_mask_move(app);
        update_button_states(app);
        gtk_label_set_text(app->status_label, "Selection moved.");
        update_canvas_cursor(app);
    } else if (app->is_panning) {
        app->is_panning = false;
        update_canvas_cursor(app);
    } else if (app->mode == AppMode::PICK_SOURCE) {
        // In source-picking mode the rect defines the clone source, not a mask
        finish_pick_source(app);
    } else if (app->mode == AppMode::PICK_DEST) {
        // In clone destination mode the rect defines the clone target mask
        finish_pick_destination(app);
    } else if (app->tool == DrawTool::RECT) {
        cv::Point p0 = {(int)app->drag_start_x, (int)app->drag_start_y};
        cv::Point p1 = {(int)app->drag_cur_x,   (int)app->drag_cur_y};
        draw_rect_on_mask(app, p0, p1);
        gtk_widget_queue_draw(GTK_WIDGET(app->canvas));
    }
    update_button_states(app);
}

// ── Middle-button drag: pan ───────────────────
static void on_pan_drag_begin(GtkGestureDrag *gesture, double x, double y, gpointer user_data) {
    AppState *app = static_cast<AppState*>(user_data);
    (void)gesture;
    if (!app->has_image) return;
    begin_move_or_pan(app, x, y);
}

static void on_pan_drag_update(GtkGestureDrag *gesture, double dx, double dy, gpointer user_data) {
    AppState *app = static_cast<AppState*>(user_data);
    on_draw_drag_update(gesture, dx, dy, user_data);
    gtk_widget_queue_draw(GTK_WIDGET(app->canvas));
}

static void on_pan_drag_end(GtkGestureDrag *gesture, double dx, double dy, gpointer user_data) {
    on_draw_drag_end(gesture, dx, dy, user_data);
}

// ── Mouse motion: track cursor for zoom-to-cursor ──
static void on_motion(GtkEventControllerMotion *ctrl, double x, double y, gpointer user_data) {
    AppState *app = static_cast<AppState*>(user_data);
    (void)ctrl;
    app->mouse_x = x;
    app->mouse_y = y;
    app->cursor_on_canvas = true;
    if (app->has_image) {
        update_canvas_cursor(app);
    }
}

static void on_motion_leave(GtkEventControllerMotion *ctrl, gpointer user_data) {
    AppState *app = static_cast<AppState*>(user_data);
    (void)ctrl;
    // Do NOT hide brush cursor — keep it visible at the last known position
    // so the user can see the brush size while adjusting the slider in the sidebar.
    // cursor_on_canvas stays true; mouse_x/mouse_y keep last position.
    (void)app;
}

// ── Scroll: zoom toward mouse cursor ─────────
static gboolean on_scroll(GtkEventControllerScroll *ctrl, double dx, double dy, gpointer user_data) {
    AppState *app = static_cast<AppState*>(user_data);
    (void)ctrl; (void)dx;
    if (!app->has_image) return TRUE;

    double old_zoom = app->zoom;
    if (dy < 0) {
        app->zoom = std::min(app->zoom * 1.12, 20.0);
    } else {
        app->zoom = std::max(app->zoom / 1.12, 0.05);
    }

    // Zoom toward mouse cursor:
    // Keep the image point under the cursor stationary.
    // image_point = (mouse - center - pan) / old_zoom
    // new_pan = mouse - center - image_point * new_zoom
    int ww = gtk_widget_get_width(GTK_WIDGET(app->canvas));
    int wh = gtk_widget_get_height(GTK_WIDGET(app->canvas));
    double cx = ww / 2.0;
    double cy = wh / 2.0;

    // image coords under cursor before zoom
    double img_x = (app->mouse_x - cx - app->pan_x) / old_zoom;
    double img_y = (app->mouse_y - cy - app->pan_y) / old_zoom;

    // new pan so that same image point stays under cursor
    app->pan_x = app->mouse_x - cx - img_x * app->zoom;
    app->pan_y = app->mouse_y - cy - img_y * app->zoom;

    update_canvas_size(app);

    char buf[32];
    snprintf(buf, sizeof(buf), "%.0f%%", app->zoom * 100.0);
    gtk_label_set_text(app->zoom_label, buf);

    gtk_widget_queue_draw(GTK_WIDGET(app->canvas));
    return TRUE;
}

// ─────────────────────────────────────────────
// Drag & Drop
// ─────────────────────────────────────────────
static gboolean drop_target_drop(GtkDropTarget *target, const GValue *value,
                                  double x, double y, gpointer user_data) {
    AppState *app = static_cast<AppState*>(user_data);
    (void)target; (void)x; (void)y;

    if (G_VALUE_HOLDS(value, G_TYPE_FILE)) {
        GFile *file = G_FILE(g_value_get_object(value));
        gchar *path = g_file_get_path(file);
        if (path) {
            load_image_from_path(app, path);
            g_free(path);
            return TRUE;
        }
    } else if (G_VALUE_HOLDS(value, GDK_TYPE_FILE_LIST)) {
        GSList *files = (GSList*)g_value_get_boxed(value);
        if (files) {
            GFile *file = G_FILE(files->data);
            gchar *path = g_file_get_path(file);
            if (path) {
                load_image_from_path(app, path);
                g_free(path);
                return TRUE;
            }
        }
    }
    return FALSE;
}

// ─────────────────────────────────────────────
// Toolbar button callbacks
// ─────────────────────────────────────────────
static void on_open_clicked(GtkButton *btn, gpointer user_data) {
    AppState *app = static_cast<AppState*>(user_data);
    (void)btn;

    GtkFileDialog *dialog = gtk_file_dialog_new();
    gtk_file_dialog_set_title(dialog, "Open Image");

    // All image types filter
    GtkFileFilter *filter = gtk_file_filter_new();
    gtk_file_filter_set_name(filter, "Slike");
    gtk_file_filter_add_mime_type(filter, "image/*");
    GListStore *filters = g_list_store_new(GTK_TYPE_FILE_FILTER);
    g_list_store_append(filters, filter);
    gtk_file_dialog_set_filters(dialog, G_LIST_MODEL(filters));
    g_object_unref(filter);
    g_object_unref(filters);

    gtk_file_dialog_open(dialog, GTK_WINDOW(app->window), nullptr,
        [](GObject *source, GAsyncResult *result, gpointer data) {
            AppState *app = static_cast<AppState*>(data);
            GError *error = nullptr;
            GFile *file = gtk_file_dialog_open_finish(
                GTK_FILE_DIALOG(source), result, &error);
            if (file) {
                gchar *path = g_file_get_path(file);
                if (path) {
                    load_image_from_path(app, path);
                    g_free(path);
                }
                g_object_unref(file);
            }
            if (error) g_error_free(error);
            g_object_unref(source);
        },
        app
    );
}

static void on_save_clicked(GtkButton *btn, gpointer user_data) {
    AppState *app = static_cast<AppState*>(user_data);
    (void)btn;
    if (!app->has_image) return;

    GtkFileDialog *dialog = gtk_file_dialog_new();
    gtk_file_dialog_set_title(dialog, "Save Image As…");

    // Set initial filename
    std::string base = std::filesystem::path(app->current_file_path).stem().string();
    std::string initial = base + "_erased.png";
    gtk_file_dialog_set_initial_name(dialog, initial.c_str());

    gtk_file_dialog_save(dialog, GTK_WINDOW(app->window), nullptr,
        [](GObject *source, GAsyncResult *result, gpointer data) {
            AppState *app = static_cast<AppState*>(data);
            GError *error = nullptr;
            GFile *file = gtk_file_dialog_save_finish(
                GTK_FILE_DIALOG(source), result, &error);
            if (file) {
                gchar *path = g_file_get_path(file);
                if (path) {
                    std::string spath(path);
                    // Ensure .png extension if none
                    if (spath.find('.') == std::string::npos) spath += ".png";

                    bool ok = cv::imwrite(spath, app->current_image);
                    if (!ok) {
                        show_error_dialog(app, "Could not save image:\n" + spath);
                    } else {
                        gtk_label_set_text(app->status_label,
                            ("Saved: " + std::filesystem::path(spath).filename().string()).c_str());
                    }
                    g_free(path);
                }
                g_object_unref(file);
            }
            if (error) g_error_free(error);
            g_object_unref(source);
        },
        app
    );
}

// Called on main thread via g_idle_add once the worker thread finishes
static gboolean erase_done_cb(gpointer user_data) {
    std::unique_ptr<EraseJobResult> job(static_cast<EraseJobResult*>(user_data));
    AppState *app = job->app;
    if (app->erase_thread.joinable()) app->erase_thread.join();

    if (!job->result.empty()) {
        app->current_image = job->result;
        clear_mask(app);
        rebuild_image_surface(app);
        gtk_label_set_text(app->status_label,
            job->status_message.empty() ? "Done." : job->status_message.c_str());
    } else {
        gtk_label_set_text(app->status_label,
            job->status_message.empty() ? "Erase failed." : job->status_message.c_str());
    }

    gtk_spinner_stop(app->spinner);
    app->is_erasing = false;

    // Re-enable controls
    gtk_widget_set_sensitive(GTK_WIDGET(app->btn_erase),      TRUE);
    gtk_widget_set_sensitive(GTK_WIDGET(app->btn_fixit),      TRUE);
    gtk_widget_set_sensitive(GTK_WIDGET(app->btn_clone_here), app->source_rect_set && app->clone_dest_set);
    gtk_widget_set_sensitive(GTK_WIDGET(app->btn_cancel_clone),
        app->source_rect_set || app->clone_dest_set ||
        app->mode == AppMode::PICK_SOURCE || app->mode == AppMode::PICK_DEST);
    gtk_widget_set_sensitive(GTK_WIDGET(app->btn_open),       TRUE);
    gtk_widget_set_sensitive(GTK_WIDGET(app->btn_undo),       !app->undo_stack.empty());
    gtk_widget_set_sensitive(GTK_WIDGET(app->btn_redo),       !app->redo_stack.empty());
    update_clone_ui(app);
    gtk_widget_queue_draw(GTK_WIDGET(app->canvas));

    return G_SOURCE_REMOVE;
}

static void do_erase(AppState *app) {
    if (!app->has_image || app->is_erasing) return;

    // Check if mask is non-empty
    if (cv::countNonZero(app->mask) == 0) {
        show_info_dialog(app, "First mark the object with the brush or rectangle selection.");
        return;
    }

    // Get method
    guint method_idx = gtk_drop_down_get_selected(app->method_dropdown);
    InpaintMethod method = (InpaintMethod)method_idx;
    if (method == InpaintMethod::DNN_LAMA && (!app->dnn_loaded || !app->ort_session)) {
        if (app->dnn_loading) {
            show_info_dialog(app, "LaMa model is still loading. Please try again in a moment.");
        } else {
            show_info_dialog(app, "LaMa model is not loaded. Check that the model file exists in the models folder.");
        }
        return;
    }

    // Save undo state (image + mask)
    app->push_undo();

    // Disable controls while working, start spinner
    app->is_erasing = true;
    gtk_spinner_start(app->spinner);
    gtk_label_set_text(app->status_label, "Removing…");
    gtk_widget_set_sensitive(GTK_WIDGET(app->btn_erase), FALSE);
    gtk_widget_set_sensitive(GTK_WIDGET(app->btn_open),  FALSE);
    gtk_widget_set_sensitive(GTK_WIDGET(app->btn_undo),  FALSE);
    gtk_widget_set_sensitive(GTK_WIDGET(app->btn_redo),  FALSE);

    // Snapshot inputs for the worker thread (cheap header copies, shared pixel data)
    cv::Mat img_snap  = app->current_image;
    cv::Mat mask_snap = app->mask.clone();

    app->erase_thread = std::thread([app, img_snap, mask_snap, method]() {
        cv::Mat result;
        std::string status_message;
        try {
            cv::Mat dilated_mask;
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5});
            cv::dilate(mask_snap, dilated_mask, kernel);

            switch (method) {
            case InpaintMethod::TELEA:
                cv::inpaint(img_snap, dilated_mask, result, 3, cv::INPAINT_TELEA);
                break;
            case InpaintMethod::NS:
                cv::inpaint(img_snap, dilated_mask, result, 3, cv::INPAINT_NS);
                break;
            case InpaintMethod::MULTISCALE:
                result = inpaint_multiscale(img_snap, dilated_mask);
                break;
            case InpaintMethod::PATCHMATCH:
                result = inpaint_patchmatch(img_snap, dilated_mask);
                break;
            case InpaintMethod::DNN_LAMA:
                result = inpaint_lama(app, img_snap, dilated_mask, &status_message);
                break;
            }
        } catch (const Ort::Exception &e) {
            g_printerr("LaMa/ONNX Runtime error during erase: %s\n", e.what());
            status_message = "Erase failed: LaMa/ONNX Runtime error.";
        } catch (const cv::Exception &e) {
            g_printerr("OpenCV error during erase: %s\n", e.what());
            status_message = "Erase failed: OpenCV error.";
        } catch (const std::exception &e) {
            g_printerr("Error during erase: %s\n", e.what());
            status_message = "Erase failed.";
        }
        auto *job = new EraseJobResult{app, result, status_message};
        g_idle_add(erase_done_cb, job);
    });
}

static void on_erase_clicked(GtkButton *btn, gpointer user_data) {
    AppState *app = static_cast<AppState*>(user_data);
    (void)btn;
    do_erase(app);
}

static void do_fixit(AppState *app) {
    if (!app->has_image || app->is_erasing) return;
    if (cv::countNonZero(app->mask) == 0) {
        show_info_dialog(app, "First mark the area with the brush or rectangle selection.");
        return;
    }

    app->push_undo();
    app->is_erasing = true;
    gtk_spinner_start(app->spinner);
    gtk_label_set_text(app->status_label, "Fixing…");
    gtk_widget_set_sensitive(GTK_WIDGET(app->btn_erase),  FALSE);
    gtk_widget_set_sensitive(GTK_WIDGET(app->btn_fixit),  FALSE);
    gtk_widget_set_sensitive(GTK_WIDGET(app->btn_open),   FALSE);
    gtk_widget_set_sensitive(GTK_WIDGET(app->btn_undo),   FALSE);
    gtk_widget_set_sensitive(GTK_WIDGET(app->btn_redo),   FALSE);

    cv::Mat img_snap  = app->current_image;
    cv::Mat mask_snap = app->mask.clone();

    app->erase_thread = std::thread([app, img_snap, mask_snap]() {
        cv::Mat result;
        std::string status_message;
        try {
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3});
            cv::Mat dilated;
            cv::dilate(mask_snap, dilated, kernel);
            result = inpaint_fixit(img_snap, dilated);
        } catch (const cv::Exception &e) {
            g_printerr("OpenCV error during Fix It: %s\n", e.what());
            status_message = "Fix It failed: OpenCV error.";
        }
        auto *job = new EraseJobResult{app, result, status_message};
        g_idle_add(erase_done_cb, job);
    });
}

static void on_fixit_clicked(GtkButton *btn, gpointer user_data) {
    AppState *app = static_cast<AppState*>(user_data);
    (void)btn;
    do_fixit(app);
}

// ── Clone Heal handlers ───────────────────────────────────────────────────

// "Cancel Clone" — reset source rect and return to normal mode
static void on_cancel_clone_clicked(GtkButton *btn, gpointer user_data) {
    AppState *app = static_cast<AppState*>(user_data);
    (void)btn;
    app->mode            = AppMode::NORMAL;
    app->source_rect_set = false;
    clear_mask(app);
    // Restore brush tool
    app->tool = DrawTool::BRUSH;
    gtk_toggle_button_set_active(app->btn_tool_rect,  FALSE);
    gtk_toggle_button_set_active(app->btn_tool_brush, TRUE);
    gtk_widget_set_sensitive(GTK_WIDGET(app->btn_clone_here),   FALSE);
    gtk_widget_set_sensitive(GTK_WIDGET(app->btn_cancel_clone), FALSE);
    update_clone_ui(app);
    gtk_label_set_text(app->status_label, "Clone cancelled.");
    gtk_widget_queue_draw(GTK_WIDGET(app->canvas));
}

// "Pick Source" — switch to source-selection mode; user draws a rect on canvas
static void on_pick_source_clicked(GtkButton *btn, gpointer user_data) {
    AppState *app = static_cast<AppState*>(user_data);
    (void)btn;
    if (!app->has_image) return;
    app->mode = AppMode::PICK_SOURCE;
    app->source_rect_set = false;
    clear_mask(app);
    // Force rect tool so the drag gesture draws the source box
    app->tool = DrawTool::RECT;
    gtk_toggle_button_set_active(app->btn_tool_brush, FALSE);
    gtk_toggle_button_set_active(app->btn_tool_rect,  TRUE);
    gtk_label_set_text(app->status_label,
        "Draw a rectangle over the source region to clone from.");
    gtk_widget_set_sensitive(GTK_WIDGET(app->btn_clone_here),   FALSE);
    gtk_widget_set_sensitive(GTK_WIDGET(app->btn_cancel_clone), TRUE);
    update_clone_ui(app);
    gtk_widget_queue_draw(GTK_WIDGET(app->canvas));
}

// Called after user finishes drawing source rect in PICK_SOURCE mode.
// drag_start/cur are already in image pixel coords (set by on_draw_drag_begin/update).
static void finish_pick_source(AppState *app) {
    int iw = app->current_image.cols, ih = app->current_image.rows;

    double ix0 = app->drag_start_x, iy0 = app->drag_start_y;
    double ix1 = app->drag_cur_x,   iy1 = app->drag_cur_y;

    if (ix0 > ix1) std::swap(ix0, ix1);
    if (iy0 > iy1) std::swap(iy0, iy1);

    int rx = std::clamp((int)ix0, 0, iw - 1);
    int ry = std::clamp((int)iy0, 0, ih - 1);
    int rw = std::clamp((int)(ix1 - ix0), 4, iw - rx);
    int rh = std::clamp((int)(iy1 - iy0), 4, ih - ry);

    if (rw >= 4 && rh >= 4) {
        app->source_rect     = cv::Rect(rx, ry, rw, rh);
        app->source_rect_set = true;
        app->clone_dest_set  = false;
        gtk_widget_set_sensitive(GTK_WIDGET(app->btn_clone_here),   FALSE);
        gtk_widget_set_sensitive(GTK_WIDGET(app->btn_cancel_clone), TRUE);
        gtk_label_set_text(app->status_label,
            "Source selected. Now draw the destination rectangle.");
    }

    // Stay in guided clone mode: the next rectangle is the destination, not erase/inpaint.
    app->mode = AppMode::PICK_DEST;
    app->tool = DrawTool::RECT;
    gtk_toggle_button_set_active(app->btn_tool_brush, FALSE);
    gtk_toggle_button_set_active(app->btn_tool_rect,  TRUE);

    update_clone_ui(app);
    gtk_widget_queue_draw(GTK_WIDGET(app->canvas));
}

// Called after user finishes drawing destination rect in PICK_DEST mode.
static void finish_pick_destination(AppState *app) {
    int iw = app->current_image.cols, ih = app->current_image.rows;

    double ix0 = app->drag_start_x, iy0 = app->drag_start_y;
    double ix1 = app->drag_cur_x,   iy1 = app->drag_cur_y;

    if (ix0 > ix1) std::swap(ix0, ix1);
    if (iy0 > iy1) std::swap(iy0, iy1);

    int rx = std::clamp((int)ix0, 0, iw - 1);
    int ry = std::clamp((int)iy0, 0, ih - 1);
    int rw = std::clamp((int)(ix1 - ix0), 4, iw - rx);
    int rh = std::clamp((int)(iy1 - iy0), 4, ih - ry);

    cv::Mat dest_mask = cv::Mat::zeros(ih, iw, CV_8UC1);
    if (rw >= 4 && rh >= 4) {
        cv::rectangle(dest_mask, {rx, ry}, {rx + rw - 1, ry + rh - 1},
                      cv::Scalar(255), -1);
        app->clone_dest_set = true;
        gtk_widget_set_sensitive(GTK_WIDGET(app->btn_clone_here), TRUE);
        gtk_label_set_text(app->status_label,
            "Destination selected. Click \"Clone Here\" to apply, or draw again to adjust.");
    } else {
        app->clone_dest_set = false;
        gtk_widget_set_sensitive(GTK_WIDGET(app->btn_clone_here), FALSE);
        gtk_label_set_text(app->status_label, "Destination rectangle is too small.");
    }
    set_mask(app, std::move(dest_mask), false);
    update_clone_ui(app);
}

static void do_clone_heal(AppState *app) {
    if (!app->has_image || app->is_erasing) return;
    if (!app->source_rect_set) {
        show_info_dialog(app, "First pick a source region with \"Pick Source\".");
        return;
    }
    if (!app->clone_dest_set || cv::countNonZero(app->mask) == 0) {
        show_info_dialog(app, "Draw the destination rectangle first.");
        return;
    }

    app->push_undo();
    app->is_erasing = true;
    app->mode = AppMode::PICK_DEST;
    gtk_spinner_start(app->spinner);
    gtk_label_set_text(app->status_label, "Cloning…");
    gtk_widget_set_sensitive(GTK_WIDGET(app->btn_erase),        FALSE);
    gtk_widget_set_sensitive(GTK_WIDGET(app->btn_fixit),        FALSE);
    gtk_widget_set_sensitive(GTK_WIDGET(app->btn_clone_here),   FALSE);
    gtk_widget_set_sensitive(GTK_WIDGET(app->btn_cancel_clone), FALSE);
    gtk_widget_set_sensitive(GTK_WIDGET(app->btn_open),         FALSE);
    gtk_widget_set_sensitive(GTK_WIDGET(app->btn_undo),         FALSE);
    gtk_widget_set_sensitive(GTK_WIDGET(app->btn_redo),         FALSE);

    cv::Mat img_snap   = app->current_image;
    cv::Mat mask_snap  = app->mask.clone();
    cv::Rect src_snap  = app->source_rect;

    app->erase_thread = std::thread([app, img_snap, mask_snap, src_snap]() {
        cv::Mat result;
        std::string status_message;
        try {
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3});
            cv::Mat dilated;
            cv::dilate(mask_snap, dilated, kernel);
            result = inpaint_clone_heal(img_snap, dilated, src_snap);
        } catch (const cv::Exception &e) {
            g_printerr("OpenCV error during Clone Heal: %s\n", e.what());
            status_message = "Clone Heal failed: OpenCV error.";
        }
        auto *job = new EraseJobResult{app, result, status_message};
        g_idle_add(erase_done_cb, job);
    });
}

static void on_clone_here_clicked(GtkButton *btn, gpointer user_data) {
    AppState *app = static_cast<AppState*>(user_data);
    (void)btn;
    do_clone_heal(app);
}

static void on_undo_clicked(GtkButton *btn, gpointer user_data) {
    AppState *app = static_cast<AppState*>(user_data);
    (void)btn;
    if (app->undo_stack.empty()) return;

    // Push current state to redo (without clearing redo — we want proper redo chain)
    app->redo_stack.push_back({app->current_image.clone(), app->mask.clone()});

    auto &snap = app->undo_stack.back();
    app->current_image = snap.image;
    set_mask(app, snap.mask.clone(), false);
    app->undo_stack.pop_back();

    rebuild_image_surface(app);
    update_button_states(app);
    gtk_label_set_text(app->status_label, "Undone.");
}

static void on_redo_clicked(GtkButton *btn, gpointer user_data) {
    AppState *app = static_cast<AppState*>(user_data);
    (void)btn;
    if (app->redo_stack.empty()) return;

    app->undo_stack.push_back({app->current_image.clone(), app->mask.clone()});

    auto &snap = app->redo_stack.back();
    app->current_image = snap.image;
    set_mask(app, snap.mask.clone(), false);
    app->redo_stack.pop_back();

    rebuild_image_surface(app);
    update_button_states(app);
    gtk_label_set_text(app->status_label, "Redone.");
}

static void on_clear_mask_clicked(GtkButton *btn, gpointer user_data) {
    AppState *app = static_cast<AppState*>(user_data);
    (void)btn;
    if (!app->has_image) return;
    clear_mask(app);
    update_button_states(app);
    if (app->mode == AppMode::PICK_DEST && app->source_rect_set)
        gtk_label_set_text(app->status_label, "Destination cleared. Draw the destination rectangle.");
    else
        gtk_label_set_text(app->status_label, "Mask cleared.");
}

static void on_tool_brush_clicked(GtkButton *btn, gpointer user_data) {
    AppState *app = static_cast<AppState*>(user_data);
    (void)btn;
    app->tool = DrawTool::BRUSH;
    gtk_toggle_button_set_active(app->btn_tool_brush, TRUE);
    gtk_toggle_button_set_active(app->btn_tool_rect,  FALSE);
}

static void on_tool_rect_clicked(GtkButton *btn, gpointer user_data) {
    AppState *app = static_cast<AppState*>(user_data);
    (void)btn;
    app->tool = DrawTool::RECT;
    gtk_toggle_button_set_active(app->btn_tool_brush, FALSE);
    gtk_toggle_button_set_active(app->btn_tool_rect,  TRUE);
    gtk_widget_queue_draw(GTK_WIDGET(app->canvas));
}

static void on_zoom_in_clicked(GtkButton *btn, gpointer user_data) {
    AppState *app = static_cast<AppState*>(user_data);
    (void)btn;
    app->zoom = std::min(app->zoom * 1.25, 20.0);
    char buf[32];
    snprintf(buf, sizeof(buf), "%.0f%%", app->zoom * 100.0);
    gtk_label_set_text(app->zoom_label, buf);
    update_canvas_size(app);
    gtk_widget_queue_draw(GTK_WIDGET(app->canvas));
}

static void on_zoom_out_clicked(GtkButton *btn, gpointer user_data) {
    AppState *app = static_cast<AppState*>(user_data);
    (void)btn;
    app->zoom = std::max(app->zoom / 1.25, 0.05);
    char buf[32];
    snprintf(buf, sizeof(buf), "%.0f%%", app->zoom * 100.0);
    gtk_label_set_text(app->zoom_label, buf);
    update_canvas_size(app);
    gtk_widget_queue_draw(GTK_WIDGET(app->canvas));
}

static void on_zoom_fit_clicked(GtkButton *btn, gpointer user_data) {
    AppState *app = static_cast<AppState*>(user_data);
    (void)btn;
    if (!app->has_image) return;

    int ww = gtk_widget_get_width(GTK_WIDGET(app->scrolled_window));
    int wh = gtk_widget_get_height(GTK_WIDGET(app->scrolled_window));
    double zx = (double)ww / app->current_image.cols;
    double zy = (double)wh / app->current_image.rows;
    app->zoom  = std::min(zx, zy) * 0.95;
    app->pan_x = 0;
    app->pan_y = 0;

    char buf[32];
    snprintf(buf, sizeof(buf), "%.0f%%", app->zoom * 100.0);
    gtk_label_set_text(app->zoom_label, buf);
    update_canvas_size(app);
    gtk_widget_queue_draw(GTK_WIDGET(app->canvas));
}

// ─────────────────────────────────────────────
// Dialogs
// ─────────────────────────────────────────────
static void show_error_dialog(AppState *app, const std::string &message) {
    AdwAlertDialog *dialog = ADW_ALERT_DIALOG(adw_alert_dialog_new("Error", message.c_str()));
    adw_alert_dialog_add_response(dialog, "ok", "OK");
    adw_dialog_present(ADW_DIALOG(dialog), GTK_WIDGET(app->window));
}

static void show_info_dialog(AppState *app, const std::string &message) {
    AdwAlertDialog *dialog = ADW_ALERT_DIALOG(adw_alert_dialog_new("Notice", message.c_str()));
    adw_alert_dialog_add_response(dialog, "ok", "OK");
    adw_dialog_present(ADW_DIALOG(dialog), GTK_WIDGET(app->window));
}

static void on_uri_launch_done(GObject *source_object, GAsyncResult *result, gpointer user_data) {
    AppState *app = static_cast<AppState*>(user_data);
    GError *error = nullptr;
    if (!gtk_uri_launcher_launch_finish(GTK_URI_LAUNCHER(source_object), result, &error)) {
        std::string message = error ? error->message : "Could not open link.";
        show_error_dialog(app, message);
    }
    if (error) g_error_free(error);
}

static void register_project_icons() {
    std::vector<std::filesystem::path> candidates;

    char exe_buf[4096] = {};
    ssize_t len = readlink("/proc/self/exe", exe_buf, sizeof(exe_buf) - 1);
    if (len > 0) {
        exe_buf[len] = '\0';
        std::filesystem::path exe_dir = std::filesystem::path(exe_buf).parent_path();
        candidates.push_back(exe_dir / "../share/icons");
        candidates.push_back(exe_dir / "../data/icons");
    }
    candidates.push_back(std::filesystem::path("data/icons"));

    GtkIconTheme *theme = gtk_icon_theme_get_for_display(gdk_display_get_default());
    for (const auto &path : candidates) {
        if (std::filesystem::exists(path / "hicolor")) {
            gtk_icon_theme_add_search_path(theme, path.string().c_str());
        }
    }
}

// ─────────────────────────────────────────────
// Keyboard shortcuts
// ─────────────────────────────────────────────
static gboolean on_key_pressed(GtkEventControllerKey *ctrl,
                                 guint keyval, guint keycode,
                                 GdkModifierType state, gpointer user_data) {
    AppState *app = static_cast<AppState*>(user_data);
    (void)ctrl; (void)keycode;

    bool ctrl_held = (state & GDK_CONTROL_MASK) != 0 || is_control_key(keyval);
    if (ctrl_held && !app->ctrl_pressed) {
        app->ctrl_pressed = true;
        update_canvas_cursor(app);
    }

    if (ctrl_held) {
        if (keyval == GDK_KEY_z) { on_undo_clicked(nullptr, app); return TRUE; }
        if (keyval == GDK_KEY_y || keyval == GDK_KEY_Z) { on_redo_clicked(nullptr, app); return TRUE; }
        if (keyval == GDK_KEY_o) { on_open_clicked(nullptr, app); return TRUE; }
        if (keyval == GDK_KEY_s) { on_save_clicked(nullptr, app); return TRUE; }
        if (keyval == GDK_KEY_Return || keyval == GDK_KEY_KP_Enter) {
            do_erase(app); return TRUE;
        }
    }
    if (keyval == GDK_KEY_Delete || keyval == GDK_KEY_BackSpace) {
        on_clear_mask_clicked(nullptr, app); return TRUE;
    }
    if (keyval == GDK_KEY_b || keyval == GDK_KEY_B) {
        on_tool_brush_clicked(nullptr, app); return TRUE;
    }
    if (keyval == GDK_KEY_r || keyval == GDK_KEY_R) {
        on_tool_rect_clicked(nullptr, app); return TRUE;
    }
    if (keyval == GDK_KEY_plus || keyval == GDK_KEY_equal) {
        on_zoom_in_clicked(nullptr, app); return TRUE;
    }
    if (keyval == GDK_KEY_minus) {
        on_zoom_out_clicked(nullptr, app); return TRUE;
    }
    if (keyval == GDK_KEY_0) {
        on_zoom_fit_clicked(nullptr, app); return TRUE;
    }
    return FALSE;
}

static gboolean on_key_released(GtkEventControllerKey *ctrl,
                                guint keyval, guint keycode,
                                GdkModifierType state, gpointer user_data) {
    AppState *app = static_cast<AppState*>(user_data);
    (void)ctrl; (void)keycode;
    bool ctrl_still_held = !is_control_key(keyval) && (state & GDK_CONTROL_MASK) != 0;
    if (app->ctrl_pressed != ctrl_still_held) {
        app->ctrl_pressed = ctrl_still_held;
        update_canvas_cursor(app);
    }
    return FALSE;
}

// ─────────────────────────────────────────────
// Brush size spinner callback
// ─────────────────────────────────────────────
static void on_brush_size_changed(GtkRange *range, gpointer user_data) {
    AppState *app = static_cast<AppState*>(user_data);
    app->brush_radius = (int)gtk_range_get_value(range);
    // Update the value label next to the slider
    if (app->spin_brush_size) {
        GtkWidget *label = static_cast<GtkWidget*>(
            g_object_get_data(G_OBJECT(app->spin_brush_size), "value-label"));
        if (label) {
            char buf[8];
            snprintf(buf, sizeof(buf), "%d", app->brush_radius);
            gtk_label_set_text(GTK_LABEL(label), buf);
        }
    }
    if (app->tool == DrawTool::BRUSH)
        gtk_widget_queue_draw(GTK_WIDGET(app->canvas));
}

// ─────────────────────────────────────────────
// App activation
// ─────────────────────────────────────────────
static void on_activate(GtkApplication *gtk_app, gpointer user_data) {
    AppState *app = static_cast<AppState*>(user_data);
    register_project_icons();

    // Find model directory — priority order:
    //  1. BAERASER_MODEL_PATH env var (set by AppRun wrapper)
    //  2. ../models/ relative to real binary (dev build layout)
    //  3. ./models/ relative to cwd (fallback)
    {
        const char *env_path = g_getenv("BAERASER_MODEL_PATH");
        if (env_path && std::filesystem::exists(env_path)) {
            app->model_path = env_path;
        } else {
            char exe_buf[4096] = {};
            ssize_t len = readlink("/proc/self/exe", exe_buf, sizeof(exe_buf)-1);
            std::string exe_dir = ".";
            if (len > 0) {
                exe_buf[len] = '\0';
                exe_dir = std::filesystem::path(exe_buf).parent_path().string();
            }
            app->model_path = exe_dir + "/../models";
            if (!std::filesystem::exists(app->model_path))
                app->model_path = "./models";
        }
    }
    // ── Window ───────────────────────────────
    app->window = ADW_APPLICATION_WINDOW(adw_application_window_new(gtk_app));
    gtk_window_set_title(GTK_WINDOW(app->window), "baEraser");
    gtk_window_set_default_size(GTK_WINDOW(app->window), 1100, 780);

    // ── Root layout: ToolbarView ──────────────
    AdwToolbarView *toolbar_view = ADW_TOOLBAR_VIEW(adw_toolbar_view_new());
    adw_application_window_set_content(app->window, GTK_WIDGET(toolbar_view));

    // ── Header bar ───────────────────────────
    AdwHeaderBar *header = ADW_HEADER_BAR(adw_header_bar_new());

    // Left side: open, save, separator, undo/redo
    GtkWidget *btn_open = gtk_button_new_from_icon_name("document-open-symbolic");
    gtk_widget_set_tooltip_text(btn_open, "Open image (Ctrl+O)");
    g_signal_connect(btn_open, "clicked", G_CALLBACK(on_open_clicked), app);
    app->btn_open = GTK_BUTTON(btn_open);
    adw_header_bar_pack_start(header, btn_open);

    GtkWidget *btn_save = gtk_button_new_from_icon_name("document-save-as-symbolic");
    gtk_widget_set_tooltip_text(btn_save, "Save as… (Ctrl+S)");
    gtk_widget_add_css_class(btn_save, "suggested-action");
    g_signal_connect(btn_save, "clicked", G_CALLBACK(on_save_clicked), app);
    app->btn_save = GTK_BUTTON(btn_save);
    adw_header_bar_pack_start(header, btn_save);

    // Separator
    GtkWidget *sep1 = gtk_separator_new(GTK_ORIENTATION_VERTICAL);
    gtk_widget_set_margin_start(sep1, 4);
    gtk_widget_set_margin_end(sep1, 4);
    adw_header_bar_pack_start(header, sep1);

    GtkWidget *btn_undo = gtk_button_new_from_icon_name("edit-undo-symbolic");
    gtk_widget_set_tooltip_text(btn_undo, "Undo (Ctrl+Z)");
    g_signal_connect(btn_undo, "clicked", G_CALLBACK(on_undo_clicked), app);
    app->btn_undo = GTK_BUTTON(btn_undo);
    adw_header_bar_pack_start(header, btn_undo);

    GtkWidget *btn_redo = gtk_button_new_from_icon_name("edit-redo-symbolic");
    gtk_widget_set_tooltip_text(btn_redo, "Redo (Ctrl+Y)");
    g_signal_connect(btn_redo, "clicked", G_CALLBACK(on_redo_clicked), app);
    app->btn_redo = GTK_BUTTON(btn_redo);
    adw_header_bar_pack_start(header, btn_redo);

    // Right side: donate | spinner
    // pack_end inserts right-to-left, so add in reverse display order

    // Donate button (rightmost)
    GtkWidget *btn_donate = gtk_button_new_from_icon_name("emblem-favorite-symbolic");
    gtk_widget_set_tooltip_text(btn_donate, "Support baEraser — Donate via PayPal");
    gtk_widget_add_css_class(btn_donate, "flat");
    g_signal_connect(btn_donate, "clicked", G_CALLBACK(+[](GtkButton *, gpointer user_data) {
        AppState *app = static_cast<AppState*>(user_data);
        GtkUriLauncher *launcher = gtk_uri_launcher_new(
            "https://www.paypal.com/donate/?hosted_button_id=4BS9ZUXJ2P7GN");
        gtk_uri_launcher_launch(launcher, GTK_WINDOW(app->window), nullptr,
                                on_uri_launch_done, app);
        g_object_unref(launcher);
    }), app);
    adw_header_bar_pack_end(header, btn_donate);

    // Spinner
    GtkWidget *spinner = gtk_spinner_new();
    app->spinner = GTK_SPINNER(spinner);
    adw_header_bar_pack_end(header, spinner);

    adw_toolbar_view_add_top_bar(toolbar_view, GTK_WIDGET(header));

    // ── Main content: horizontal split ───────
    GtkWidget *main_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    adw_toolbar_view_set_content(toolbar_view, main_box);

    // ── Sidebar (tools panel) ─────────────────
    // Keep this panel stable: controls must wrap/clip inside it, never resize it.
    constexpr int SIDEBAR_OUTER_WIDTH = 264;
    constexpr int SIDEBAR_WIDTH = 248;
    constexpr int SIDEBAR_CONTROL_WIDTH = 224;
    constexpr int SIDEBAR_TEXT_CHARS = 24;
    auto make_sidebar_label = [=](GtkWidget *label) {
        gtk_label_set_wrap(GTK_LABEL(label), TRUE);
        gtk_label_set_wrap_mode(GTK_LABEL(label), PANGO_WRAP_WORD_CHAR);
        gtk_label_set_max_width_chars(GTK_LABEL(label), SIDEBAR_TEXT_CHARS);
        gtk_label_set_xalign(GTK_LABEL(label), 0.0f);
        gtk_widget_set_halign(label, GTK_ALIGN_FILL);
        gtk_widget_set_hexpand(label, TRUE);
    };
    GtkWidget *sidebar_shell = gtk_scrolled_window_new();
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(sidebar_shell),
                                   GTK_POLICY_NEVER, GTK_POLICY_AUTOMATIC);
    gtk_scrolled_window_set_propagate_natural_width(GTK_SCROLLED_WINDOW(sidebar_shell), FALSE);
    gtk_widget_set_hexpand(sidebar_shell, FALSE);
    gtk_widget_set_vexpand(sidebar_shell, TRUE);
    gtk_widget_set_size_request(sidebar_shell, SIDEBAR_OUTER_WIDTH, -1);
    gtk_box_append(GTK_BOX(main_box), sidebar_shell);

    GtkWidget *sidebar = gtk_box_new(GTK_ORIENTATION_VERTICAL, 8);
    gtk_widget_add_css_class(sidebar, "sidebar");
    gtk_widget_set_hexpand(sidebar, FALSE);
    gtk_widget_set_vexpand(sidebar, TRUE);
    gtk_widget_set_size_request(sidebar, SIDEBAR_WIDTH, -1);
    gtk_widget_set_margin_top(sidebar, 12);
    gtk_widget_set_margin_bottom(sidebar, 12);
    gtk_widget_set_margin_start(sidebar, 8);
    gtk_widget_set_margin_end(sidebar, 8);
    gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(sidebar_shell), sidebar);

    // Separator
    GtkWidget *sep_vert = gtk_separator_new(GTK_ORIENTATION_VERTICAL);
    gtk_box_append(GTK_BOX(main_box), sep_vert);

    // Tool group label
    GtkWidget *lbl_tools = gtk_label_new("Tools");
    gtk_widget_add_css_class(lbl_tools, "heading");
    make_sidebar_label(lbl_tools);
    gtk_box_append(GTK_BOX(sidebar), lbl_tools);

    // Tool toggle buttons
    GtkWidget *tool_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 4);
    gtk_widget_add_css_class(tool_box, "linked");

    GtkWidget *btn_brush = gtk_toggle_button_new();
    GtkWidget *brush_icon = gtk_image_new_from_icon_name("baeraser-brush-symbolic");
    gtk_button_set_child(GTK_BUTTON(btn_brush), brush_icon);
    gtk_widget_set_tooltip_text(btn_brush, "Brush (B)");
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(btn_brush), TRUE);
    g_signal_connect(btn_brush, "clicked", G_CALLBACK(on_tool_brush_clicked), app);
    app->btn_tool_brush = GTK_TOGGLE_BUTTON(btn_brush);
    gtk_box_append(GTK_BOX(tool_box), btn_brush);

    GtkWidget *btn_rect = gtk_toggle_button_new();
    GtkWidget *rect_icon = gtk_image_new_from_icon_name("baeraser-rectangle-symbolic");
    gtk_button_set_child(GTK_BUTTON(btn_rect), rect_icon);
    gtk_widget_set_tooltip_text(btn_rect, "Rectangle selection (R)");
    g_signal_connect(btn_rect, "clicked", G_CALLBACK(on_tool_rect_clicked), app);
    app->btn_tool_rect = GTK_TOGGLE_BUTTON(btn_rect);
    gtk_box_append(GTK_BOX(tool_box), btn_rect);

    gtk_box_append(GTK_BOX(sidebar), tool_box);

    // Brush size — slider row: label + value on the right
    GtkWidget *brush_hdr = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    GtkWidget *lbl_brush = gtk_label_new("Brush size");
    make_sidebar_label(lbl_brush);
    gtk_widget_set_hexpand(lbl_brush, TRUE);
    GtkWidget *lbl_brush_val = gtk_label_new("20");
    gtk_widget_add_css_class(lbl_brush_val, "dim-label");
    gtk_widget_set_margin_end(lbl_brush_val, 2);
    gtk_box_append(GTK_BOX(brush_hdr), lbl_brush);
    gtk_box_append(GTK_BOX(brush_hdr), lbl_brush_val);
    gtk_box_append(GTK_BOX(sidebar), brush_hdr);

    GtkWidget *scale = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, 1, 200, 1);
    gtk_scale_set_draw_value(GTK_SCALE(scale), FALSE);  // no number on the slider
    gtk_range_set_value(GTK_RANGE(scale), 20);
    gtk_widget_set_hexpand(scale, TRUE);
    // Store pointer to value label so the callback can update it
    g_object_set_data(G_OBJECT(scale), "value-label", lbl_brush_val);
    g_signal_connect(scale, "value-changed", G_CALLBACK(on_brush_size_changed), app);
    app->spin_brush_size = scale;
    gtk_box_append(GTK_BOX(sidebar), scale);

    // Separator
    GtkWidget *sep2 = gtk_separator_new(GTK_ORIENTATION_HORIZONTAL);
    gtk_box_append(GTK_BOX(sidebar), sep2);

    // Method label
    GtkWidget *lbl_method = gtk_label_new("Inpainting method");
    gtk_widget_add_css_class(lbl_method, "heading");
    make_sidebar_label(lbl_method);
    gtk_box_append(GTK_BOX(sidebar), lbl_method);

    // Method dropdown
    const char *methods[] = {
        "TELEA (fast)",
        "Navier-Stokes",
        "Multi-scale",
        "PatchMatch",
        "LaMa AI",
        nullptr
    };
    GtkStringList *method_list = gtk_string_list_new(methods);
    GtkDropDown *method_dd = GTK_DROP_DOWN(gtk_drop_down_new(G_LIST_MODEL(method_list), nullptr));
    gtk_drop_down_set_selected(method_dd, 3);
    gtk_widget_set_hexpand(GTK_WIDGET(method_dd), FALSE);
    gtk_widget_set_size_request(GTK_WIDGET(method_dd), SIDEBAR_CONTROL_WIDTH, -1);
    gtk_widget_set_tooltip_text(GTK_WIDGET(method_dd),
        "Choose inpainting method. PatchMatch is good for textured backgrounds; LaMa AI is best quality when loaded.");
    app->method_dropdown = method_dd;
    gtk_box_append(GTK_BOX(sidebar), GTK_WIDGET(method_dd));

    // LaMa model status label — updated async by lama_load_done_cb
    GtkWidget *lbl_dnn_info = gtk_label_new("LaMa: loading…");
    gtk_widget_add_css_class(lbl_dnn_info, "dim-label");
    make_sidebar_label(lbl_dnn_info);
    gtk_box_append(GTK_BOX(sidebar), lbl_dnn_info);
    app->lbl_model_status = GTK_LABEL(lbl_dnn_info);

    // Load LaMa model after the related widgets exist, so status and default
    // method selection can be updated reliably from the completion callback.
    load_lama_model_async(app);

    // Separator
    GtkWidget *sep3 = gtk_separator_new(GTK_ORIENTATION_HORIZONTAL);
    gtk_box_append(GTK_BOX(sidebar), sep3);

    // Clear mask + Erase Object + Fix It — side by side, linked style
    GtkWidget *action_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    gtk_widget_add_css_class(action_box, "linked");

    GtkWidget *btn_clear = gtk_button_new_from_icon_name("baeraser-clear-mask-symbolic");
    gtk_widget_set_tooltip_text(btn_clear, "Clear mask (Del)");
    gtk_widget_set_hexpand(btn_clear, TRUE);
    g_signal_connect(btn_clear, "clicked", G_CALLBACK(on_clear_mask_clicked), app);
    app->btn_clear_mask = GTK_BUTTON(btn_clear);
    gtk_box_append(GTK_BOX(action_box), btn_clear);

    GtkWidget *btn_erase = gtk_button_new_from_icon_name("baeraser-erase-object-symbolic");
    gtk_widget_add_css_class(btn_erase, "destructive-action");
    gtk_widget_set_tooltip_text(btn_erase, "Erase Object (Ctrl+Enter)");
    gtk_widget_set_hexpand(btn_erase, TRUE);
    g_signal_connect(btn_erase, "clicked", G_CALLBACK(on_erase_clicked), app);
    app->btn_erase = GTK_BUTTON(btn_erase);
    gtk_box_append(GTK_BOX(action_box), btn_erase);

    // Fix It button — auto-fills from surrounding region
    GtkWidget *btn_fixit = gtk_button_new_from_icon_name("baeraser-fixit-symbolic");
    gtk_widget_set_tooltip_text(btn_fixit,
        "Fill marked area with surrounding texture (auto-detect source)");
    gtk_widget_add_css_class(btn_fixit, "suggested-action");
    gtk_widget_set_hexpand(btn_fixit, TRUE);
    g_signal_connect(btn_fixit, "clicked", G_CALLBACK(on_fixit_clicked), app);
    app->btn_fixit = GTK_BUTTON(btn_fixit);
    gtk_box_append(GTK_BOX(action_box), btn_fixit);

    gtk_box_append(GTK_BOX(sidebar), action_box);

    // Separator before clone-heal section
    GtkWidget *sep_clone = gtk_separator_new(GTK_ORIENTATION_HORIZONTAL);
    gtk_box_append(GTK_BOX(sidebar), sep_clone);

    // Clone Heal: guided source -> destination -> apply workflow
    GtkWidget *lbl_clone = gtk_label_new("Clone Heal");
    gtk_widget_add_css_class(lbl_clone, "heading");
    make_sidebar_label(lbl_clone);
    gtk_box_append(GTK_BOX(sidebar), lbl_clone);

    GtkWidget *clone_card = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
    gtk_widget_add_css_class(clone_card, "card");
    gtk_widget_set_hexpand(clone_card, FALSE);
    gtk_widget_set_size_request(clone_card, SIDEBAR_CONTROL_WIDTH, -1);
    gtk_widget_set_margin_top(clone_card, 2);
    gtk_widget_set_margin_bottom(clone_card, 2);
    gtk_widget_set_margin_start(clone_card, 0);
    gtk_widget_set_margin_end(clone_card, 0);
    gtk_box_append(GTK_BOX(sidebar), clone_card);

    GtkWidget *lbl_clone_hint = gtk_label_new("Clone Heal copies texture from one rectangle into another.");
    gtk_widget_add_css_class(lbl_clone_hint, "dim-label");
    make_sidebar_label(lbl_clone_hint);
    app->lbl_clone_hint = GTK_LABEL(lbl_clone_hint);
    gtk_box_append(GTK_BOX(clone_card), lbl_clone_hint);

    GtkWidget *lbl_clone_source = gtk_label_new("1. Source: not selected");
    make_sidebar_label(lbl_clone_source);
    app->lbl_clone_source = GTK_LABEL(lbl_clone_source);
    gtk_box_append(GTK_BOX(clone_card), lbl_clone_source);

    GtkWidget *lbl_clone_dest = gtk_label_new("2. Destination: not selected");
    make_sidebar_label(lbl_clone_dest);
    app->lbl_clone_dest = GTK_LABEL(lbl_clone_dest);
    gtk_box_append(GTK_BOX(clone_card), lbl_clone_dest);

    GtkWidget *clone_actions = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    gtk_widget_add_css_class(clone_actions, "linked");

    GtkWidget *btn_pick = gtk_button_new_from_icon_name("baeraser-clone-pick-source-symbolic");
    gtk_widget_set_tooltip_text(btn_pick,
        "Step 1: draw a rectangle over the clean region to copy from");
    gtk_widget_set_hexpand(btn_pick, TRUE);
    g_signal_connect(btn_pick, "clicked", G_CALLBACK(on_pick_source_clicked), app);
    app->btn_pick_source = GTK_BUTTON(btn_pick);
    gtk_box_append(GTK_BOX(clone_actions), btn_pick);

    GtkWidget *btn_clone = gtk_button_new_from_icon_name("baeraser-clone-here-symbolic");
    gtk_widget_set_tooltip_text(btn_clone,
        "Step 3: blend the source rectangle into the destination rectangle");
    gtk_widget_add_css_class(btn_clone, "suggested-action");
    gtk_widget_set_sensitive(btn_clone, FALSE);
    gtk_widget_set_hexpand(btn_clone, TRUE);
    g_signal_connect(btn_clone, "clicked", G_CALLBACK(on_clone_here_clicked), app);
    app->btn_clone_here = GTK_BUTTON(btn_clone);
    gtk_box_append(GTK_BOX(clone_actions), btn_clone);

    GtkWidget *btn_cancel_clone = gtk_button_new_from_icon_name("baeraser-clone-cancel-symbolic");
    gtk_widget_set_tooltip_text(btn_cancel_clone, "Cancel clone-heal and clear the source region");
    gtk_widget_set_sensitive(btn_cancel_clone, FALSE);
    gtk_widget_set_hexpand(btn_cancel_clone, TRUE);
    g_signal_connect(btn_cancel_clone, "clicked", G_CALLBACK(on_cancel_clone_clicked), app);
    app->btn_cancel_clone = GTK_BUTTON(btn_cancel_clone);
    gtk_box_append(GTK_BOX(clone_actions), btn_cancel_clone);

    gtk_box_append(GTK_BOX(clone_card), clone_actions);

    // Spacer
    GtkWidget *spacer = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_widget_set_vexpand(spacer, TRUE);
    gtk_box_append(GTK_BOX(sidebar), spacer);

    // Zoom controls
    GtkWidget *lbl_zoom_title = gtk_label_new("Zoom");
    gtk_widget_add_css_class(lbl_zoom_title, "heading");
    make_sidebar_label(lbl_zoom_title);
    gtk_box_append(GTK_BOX(sidebar), lbl_zoom_title);

    GtkWidget *zoom_row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
    gtk_widget_set_size_request(zoom_row, SIDEBAR_CONTROL_WIDTH, -1);

    GtkWidget *zoom_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 4);
    gtk_widget_add_css_class(zoom_box, "linked");

    GtkWidget *btn_zoom_out = gtk_button_new_from_icon_name("zoom-out-symbolic");
    gtk_widget_set_tooltip_text(btn_zoom_out, "Zoom out (-)");
    g_signal_connect(btn_zoom_out, "clicked", G_CALLBACK(on_zoom_out_clicked), app);
    gtk_box_append(GTK_BOX(zoom_box), btn_zoom_out);

    GtkWidget *btn_zoom_fit = gtk_button_new_from_icon_name("zoom-fit-best-symbolic");
    gtk_widget_set_tooltip_text(btn_zoom_fit, "Fit to window (0)");
    g_signal_connect(btn_zoom_fit, "clicked", G_CALLBACK(on_zoom_fit_clicked), app);
    gtk_box_append(GTK_BOX(zoom_box), btn_zoom_fit);

    GtkWidget *btn_zoom_in = gtk_button_new_from_icon_name("zoom-in-symbolic");
    gtk_widget_set_tooltip_text(btn_zoom_in, "Zoom in (+)");
    g_signal_connect(btn_zoom_in, "clicked", G_CALLBACK(on_zoom_in_clicked), app);
    gtk_box_append(GTK_BOX(zoom_box), btn_zoom_in);

    GtkWidget *zoom_label = gtk_label_new("100%");
    gtk_widget_add_css_class(zoom_label, "dim-label");
    gtk_widget_set_hexpand(zoom_label, TRUE);
    gtk_widget_set_halign(zoom_label, GTK_ALIGN_CENTER);
    app->zoom_label = GTK_LABEL(zoom_label);

    gtk_box_append(GTK_BOX(zoom_row), zoom_box);
    gtk_box_append(GTK_BOX(zoom_row), zoom_label);
    gtk_box_append(GTK_BOX(sidebar), zoom_row);

    // ── Canvas area ───────────────────────────
    GtkWidget *canvas_container = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_widget_set_hexpand(canvas_container, TRUE);
    gtk_widget_set_vexpand(canvas_container, TRUE);
    gtk_box_append(GTK_BOX(main_box), canvas_container);

    GtkWidget *scrolled = gtk_scrolled_window_new();
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(scrolled),
                                   GTK_POLICY_NEVER, GTK_POLICY_NEVER);
    gtk_widget_set_hexpand(scrolled, TRUE);
    gtk_widget_set_vexpand(scrolled, TRUE);
    app->scrolled_window = scrolled;
    gtk_box_append(GTK_BOX(canvas_container), scrolled);

    GtkWidget *canvas = gtk_drawing_area_new();
    gtk_drawing_area_set_content_width(GTK_DRAWING_AREA(canvas), 1);
    gtk_drawing_area_set_content_height(GTK_DRAWING_AREA(canvas), 1);
    gtk_drawing_area_set_draw_func(GTK_DRAWING_AREA(canvas), canvas_draw, app, nullptr);
    app->canvas = GTK_DRAWING_AREA(canvas);
    gtk_widget_set_hexpand(canvas, TRUE);
    gtk_widget_set_vexpand(canvas, TRUE);
    gtk_widget_set_focusable(canvas, TRUE);
    gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(scrolled), canvas);

    // Left-button gesture: draw brush / rect
    GtkGestureDrag *draw_drag = GTK_GESTURE_DRAG(gtk_gesture_drag_new());
    gtk_gesture_single_set_button(GTK_GESTURE_SINGLE(draw_drag), 1); // left only
    g_signal_connect(draw_drag, "drag-begin",  G_CALLBACK(on_draw_drag_begin),  app);
    g_signal_connect(draw_drag, "drag-update", G_CALLBACK(on_draw_drag_update), app);
    g_signal_connect(draw_drag, "drag-end",    G_CALLBACK(on_draw_drag_end),    app);
    gtk_widget_add_controller(canvas, GTK_EVENT_CONTROLLER(draw_drag));

    // Middle-button gesture: pan
    GtkGestureDrag *pan_drag = GTK_GESTURE_DRAG(gtk_gesture_drag_new());
    gtk_gesture_single_set_button(GTK_GESTURE_SINGLE(pan_drag), 2); // middle only
    g_signal_connect(pan_drag, "drag-begin",  G_CALLBACK(on_pan_drag_begin),  app);
    g_signal_connect(pan_drag, "drag-update", G_CALLBACK(on_pan_drag_update), app);
    g_signal_connect(pan_drag, "drag-end",    G_CALLBACK(on_pan_drag_end),    app);
    gtk_widget_add_controller(canvas, GTK_EVENT_CONTROLLER(pan_drag));

    // Motion: track cursor for zoom-to-cursor and brush ring
    GtkEventControllerMotion *motion_ctrl = GTK_EVENT_CONTROLLER_MOTION(
        gtk_event_controller_motion_new());
    g_signal_connect(motion_ctrl, "motion", G_CALLBACK(on_motion), app);
    g_signal_connect(motion_ctrl, "leave",  G_CALLBACK(on_motion_leave), app);
    gtk_widget_add_controller(canvas, GTK_EVENT_CONTROLLER(motion_ctrl));

    // Cursor starts as default (no image loaded yet).
    // Switched to "none" in load_image_from_path() so we draw our own brush ring.

    // Scroll: zoom toward cursor
    GtkEventControllerScroll *scroll_ctrl = GTK_EVENT_CONTROLLER_SCROLL(
        gtk_event_controller_scroll_new(GTK_EVENT_CONTROLLER_SCROLL_VERTICAL));
    g_signal_connect(scroll_ctrl, "scroll", G_CALLBACK(on_scroll), app);
    gtk_widget_add_controller(canvas, GTK_EVENT_CONTROLLER(scroll_ctrl));

    // Keyboard
    GtkEventControllerKey *key_ctrl = GTK_EVENT_CONTROLLER_KEY(
        gtk_event_controller_key_new());
    gtk_event_controller_set_propagation_phase(GTK_EVENT_CONTROLLER(key_ctrl),
                                               GTK_PHASE_CAPTURE);
    g_signal_connect(key_ctrl, "key-pressed", G_CALLBACK(on_key_pressed), app);
    g_signal_connect(key_ctrl, "key-released", G_CALLBACK(on_key_released), app);
    gtk_widget_add_controller(GTK_WIDGET(app->window), GTK_EVENT_CONTROLLER(key_ctrl));

    // Drag & Drop
    GtkDropTarget *drop_target = gtk_drop_target_new(G_TYPE_INVALID, GDK_ACTION_COPY);
    GType types[] = { G_TYPE_FILE, GDK_TYPE_FILE_LIST };
    gtk_drop_target_set_gtypes(drop_target, types, 2);
    g_signal_connect(drop_target, "drop", G_CALLBACK(drop_target_drop), app);
    gtk_widget_add_controller(GTK_WIDGET(app->window), GTK_EVENT_CONTROLLER(drop_target));

    // ── Status bar ────────────────────────────
    GtkWidget *status_bar = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
    gtk_widget_add_css_class(status_bar, "toolbar");
    gtk_widget_set_margin_start(status_bar, 8);
    gtk_widget_set_margin_end(status_bar, 8);
    gtk_widget_set_margin_top(status_bar, 4);
    gtk_widget_set_margin_bottom(status_bar, 4);

    GtkWidget *status_label = gtk_label_new("Open or drop an image…");
    gtk_widget_add_css_class(status_label, "dim-label");
    gtk_widget_set_halign(status_label, GTK_ALIGN_START);
    app->status_label = GTK_LABEL(status_label);
    gtk_box_append(GTK_BOX(status_bar), status_label);

    adw_toolbar_view_add_bottom_bar(toolbar_view, status_bar);

    // ── Initial button states ─────────────────
    update_button_states(app);

    gtk_window_present(GTK_WINDOW(app->window));
}

// ─────────────────────────────────────────────
// main
// ─────────────────────────────────────────────
int main(int argc, char *argv[]) {
    AppState app;

    AdwApplication *adw_app = adw_application_new(
        "si.generacija.baEraser", G_APPLICATION_DEFAULT_FLAGS);
    g_signal_connect(adw_app, "activate", G_CALLBACK(on_activate), &app);

    int status = g_application_run(G_APPLICATION(adw_app), argc, argv);
    g_object_unref(adw_app);
    return status;
}
