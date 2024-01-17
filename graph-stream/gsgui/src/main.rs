use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use clap::Parser;
use comfy::*;
use layout::{
    core::{base::Orientation, geometry::Point, style::StyleAttr},
    std_shapes::shapes::{Arrow, Element, ShapeKind},
    topo::layout::VisualGraph,
};

#[derive(Parser)]
struct CliArgs {
    #[arg()]
    input: PathBuf,
}

fn to_mq(p: layout::core::geometry::Point) -> Vec2 {
    Vec2::new(p.x as f32, -p.y as f32)
}

struct Render {
    zoom: f32,
}

impl Render {
    fn line_size(&self) -> f32 {
        self.zoom / 512.0
    }
}

impl layout::core::format::RenderBackend for Render {
    fn draw_rect(
        &mut self,
        xy: layout::core::geometry::Point,
        size: layout::core::geometry::Point,
        _look: &layout::core::style::StyleAttr,
        _clip: Option<layout::core::format::ClipHandle>,
    ) {
        let mut size = to_mq(size);
        let mut xy = to_mq(xy);
        xy -= size * 0.5;
        size *= 2.0;

        draw_rect_outline(xy, size, self.line_size(), BLACK, 0);
        //draw_rectangle(xy.x as _, xy.y as _, size.x as _, size.y as _, BLACK);
    }

    fn draw_line(
        &mut self,
        start: layout::core::geometry::Point,
        stop: layout::core::geometry::Point,
        _look: &layout::core::style::StyleAttr,
    ) {
        draw_line(to_mq(start), to_mq(stop), self.line_size(), BLACK, 0);
    }

    fn draw_circle(
        &mut self,
        xy: layout::core::geometry::Point,
        size: layout::core::geometry::Point,
        _look: &layout::core::style::StyleAttr,
    ) {
        let size = to_mq(size);
        let mut xy = to_mq(xy);
        xy -= size * 0.5;

        draw_rect_outline(xy, size, self.line_size(), BLACK, 0);
        //draw_circle_lines(xy.x as _, xy.y as _, size.x as _, self.line_size(), BLACK);
    }

    fn draw_text(
        &mut self,
        xy: layout::core::geometry::Point,
        text: &str,
        look: &layout::core::style::StyleAttr,
    ) {
        let size = look.font_size as f32 / self.zoom * 512.0;
        if size > 8.0 {
            let xy = to_mq(xy);
            let text_params = TextParams {
                font: epaint::FontId {
                    size,
                    family: epaint::FontFamily::Proportional,
                },
                rotation: 0.0,
                color: BLACK,
            };
            draw_text_ex(text, xy, TextAlign::Center, text_params);
        }
        //let font_size = look.font_size as _;
        //let font_scale = 1.0;
        //let font = None;

        //let params = TextParams {
        //    font,
        //    font_size,
        //    font_scale,
        //    font_scale_aspect: 1.0,
        //    rotation: 0.0,
        //    color: BLACK,
        //};
        //let size = measure_text(text, font, font_size, font_scale);
        //draw_text_ex(
        //    text,
        //    xy.x as f32 - size.width * 0.5,
        //    xy.y as f32 - size.height * 0.5 + size.offset_y,
        //    params,
        //);
    }

    fn draw_arrow(
        &mut self,
        path: &[(layout::core::geometry::Point, layout::core::geometry::Point)],
        _dashed: bool,
        _head: (bool, bool),
        _look: &layout::core::style::StyleAttr,
        _text: &str,
    ) {
        // First the first point, position and control point are swapped:
        let s = path[0].0;
        let e = path[1].1;
        draw_line(to_mq(s), to_mq(e), self.line_size(), BLACK, 0);

        for l in path[1..].windows(2) {
            let s = l[0].1;
            let e = l[1].1;
            draw_line(to_mq(s), to_mq(e), self.line_size(), BLACK, 0);
        }
    }

    fn create_clip(
        &mut self,
        _xy: layout::core::geometry::Point,
        _size: layout::core::geometry::Point,
        _rounded_px: usize,
    ) -> layout::core::format::ClipHandle {
        todo!()
    }
}

#[derive(Default)]
struct Graph {
    nodes: HashMap<u64, String>,
    edges: HashSet<gs_core::Edge>,
}

impl Graph {
    fn apply(&mut self, e: gs_core::Event) {
        match e {
            gs_core::Event::AddNode(n) => assert!(
                self.nodes.insert(n.id, n.label.clone()).is_none(),
                "Add {:?}",
                n
            ),
            gs_core::Event::RemoveNode(n) => {
                assert!(self.nodes.remove(&n.id).is_some(), "Remove {:?}", n)
            }
            gs_core::Event::AddEdge(e) => {
                assert!(self.nodes.contains_key(&e.from), "Add {:?}", e);
                assert!(self.nodes.contains_key(&e.to), "Add {:?}", e);
                assert!(self.edges.insert(e.clone()), "Add {:?}", e);
            }
            gs_core::Event::RemoveEdge(e) => {
                assert!(self.nodes.contains_key(&e.from), "Remove {:?}", e);
                assert!(self.nodes.contains_key(&e.to), "Remove {:?}", e);
                assert!(self.edges.remove(&e), "Remove {:?}", e)
            }
        }
    }

    fn to_vg(&self) -> Option<VisualGraph> {
        if self.nodes.is_empty() {
            None
        } else {
            let mut out = VisualGraph::new(Orientation::TopToBottom);
            let look = StyleAttr::simple();
            let mut node_map = HashMap::new();
            //println!("{:?}", self.nodes);
            for n in &self.nodes {
                let sp0 = ShapeKind::new_box(n.1);

                //let size = measure_text(n.1, None, 10, 1.0);
                //let sz = Point::new(size.width as _, size.height as _);
                let sz = Point::new(5.0, 1.0);

                let elm = Element::create(sp0, look.clone(), Orientation::LeftToRight, sz);
                let handle = out.add_node(elm);
                node_map.insert(n.0, handle);
            }

            for e in &self.edges {
                let arrow = Arrow::simple("");
                out.add_edge(arrow, node_map[&e.from], node_map[&e.to]);
            }

            Some(out)
        }
    }
}

struct GraphTimeline {
    current_graph: Graph,
    current_step: usize,
    events: Vec<gs_core::Event>,
}

impl GraphTimeline {
    fn new(events: gs_core::EventStream) -> Self {
        let s = Self {
            current_graph: Default::default(),
            current_step: 0,
            events: events.0,
        };
        s
    }

    fn next(&mut self) {
        if self.current_step >= self.events.len() {
            return;
        }

        let e = &self.events[self.current_step];
        self.current_step += 1;
        self.current_graph.apply(e.clone());
    }

    fn prev(&mut self) {
        if self.current_step == 0 {
            return;
        }

        self.current_step -= 1;
        let e = &self.events[self.current_step];
        self.current_graph.apply(e.clone().inverse());
    }

    fn go_to(&mut self, i: usize) {
        assert!(i <= self.events.len());
        match i.cmp(&self.current_step) {
            std::cmp::Ordering::Less => {
                while self.current_step != i {
                    self.prev()
                }
            }
            std::cmp::Ordering::Equal => {}
            std::cmp::Ordering::Greater => {
                while self.current_step != i {
                    self.next()
                }
            }
        }
    }
}

// As in other state-based example we define a global state object
// for our game.
pub struct MyGame {
    timeline: GraphTimeline,
    prev_mouse: Vec2,
}

impl MyGame {
    fn make(timeline: GraphTimeline) -> Self {
        Self {
            timeline,
            prev_mouse: Vec2::new(0.0, 0.0),
        }
    }
}

// Everything interesting happens here.
impl GameLoop for MyGame {
    fn new(_c: &mut EngineState) -> Self {
        todo!()
    }

    fn update(&mut self, _c: &mut EngineContext) {
        let mut time_step = self.timeline.current_step;
        egui::Window::new("Timeline")
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(egui(), |ui| {
                ui.add(
                    egui::Slider::new(&mut time_step, 0..=self.timeline.events.len())
                        .text("Time step"),
                );
            });

        self.timeline.go_to(time_step);

        {
            let mut camera = main_camera_mut();
            let m = mouse_screen();

            if is_mouse_button_down(MouseButton::Left) {
                let d = camera.screen_to_world(self.prev_mouse) - camera.screen_to_world(m);
                camera.center += d;
            }
            let zoom_factor = (-mouse_wheel().1 * 0.05).exp();
            camera.zoom *= zoom_factor;
            //camera.zoom *= zoom_factor;
            self.prev_mouse = m;
        }

        if is_key_pressed(KeyCode::N) {
            self.timeline.next();
        }
        if is_key_pressed(KeyCode::P) {
            self.timeline.prev();
        }
        if is_key_pressed(KeyCode::R) {
            self.timeline.go_to(0)
        }
        if is_key_pressed(KeyCode::G) {
            self.timeline.go_to(self.timeline.events.len())
        }

        draw_rect(Vec2 { x: 0.0, y: 0.0 }, Vec2 { x: 10.0, y: 5.0 }, GREEN, -1);

        //set_camera(&camera);

        clear_background(WHITE);

        if let Some(mut vg) = self.timeline.current_graph.to_vg() {
            let mut r = Render {
                zoom: main_camera().zoom,
            };

            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                vg.do_it(false, true, false, &mut r)
            }));

            if let Err(err) = res {
                println!("Error in layout: {:?}", err);
            }
        }
    }
}

pub fn _comfy_default_config(config: GameConfig) -> GameConfig {
    config
}

pub async fn run() {
    // comfy includes a `define_versions!()` macro that creates a `version_str()`
    // function that returns a version from cargo & git.
    init_game_config("Graph Viewer".to_string(), "v0.0.1", _comfy_default_config);

    let engine = EngineState::new();

    let options = CliArgs::parse();

    let events = gs_core::EventStream::load(&options.input);

    let mut timeline = GraphTimeline::new(events);
    timeline.go_to(100);

    let game = MyGame::make(timeline);

    run_comfy_main_async(game, engine).await;
}

fn main() {
    pollster::block_on(run());
}
