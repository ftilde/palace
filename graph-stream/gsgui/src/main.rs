use std::{collections::BTreeMap, path::PathBuf};

use clap::Parser;
use comfy::*;
use gs_core::Timestamp;
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

struct RenderStuff {
    event_index: usize,
    layout_config: LayoutConfig,
    rects: Vec<RenderRect>,
    lines: Vec<RenderLine>,
    circles: Vec<RenderCircle>,
    texts: Vec<RenderText>,
    arrows: Vec<RenderArrow>,
    ll: Vec2,
    ur: Vec2,
}

#[derive(Copy, Clone, PartialEq, Eq)]
struct LayoutConfig {
    layout_arrange: bool,
    layout_opt: bool,
}

impl RenderStuff {
    fn empty_for_ts(event_index: usize, layout_config: LayoutConfig) -> Self {
        Self {
            event_index,
            layout_config,
            rects: Default::default(),
            lines: Default::default(),
            circles: Default::default(),
            texts: Default::default(),
            arrows: Default::default(),
            ll: Vec2::splat(f32::INFINITY),
            ur: Vec2::splat(-f32::INFINITY),
        }
    }
    fn update_bounds(&mut self, p: Vec2) {
        self.ll = self.ll.min(p);
        self.ur = self.ur.max(p);
    }
    fn region_center_and_size(&self) -> Option<(Vec2, Vec2)> {
        if self.ll.is_finite() {
            Some(((self.ll + self.ur) * 0.5, self.ur - self.ll))
        } else {
            None
        }
    }
    fn draw(&self, zoom: f32) {
        let line_size = zoom / 512.0;

        for rect in &self.rects {
            draw_rect_outline(rect.xy, rect.size, 2.0 * line_size, rect.color, 0);
        }

        for line in &self.lines {
            draw_line(line.start, line.stop, line_size, BLACK, 0);
        }

        for circle in &self.circles {
            draw_rect_outline(circle.xy, circle.size, line_size, BLACK, 0);
        }

        for text in &self.texts {
            let text_size = text.font_size as f32 / zoom * 500.0;
            if text_size > 8.0 {
                let text_params = TextParams {
                    font: epaint::FontId {
                        size: text_size,
                        family: epaint::FontFamily::Proportional,
                    },
                    rotation: 0.0,
                    color: BLACK,
                };
                draw_text_ex(&text.text, text.xy, TextAlign::Center, text_params);
            }
        }

        for arrow in &self.arrows {
            for line in arrow.points.windows(2) {
                draw_line(line[0], line[1], line_size, BLACK, 0);
            }
        }
    }
}

struct RenderRect {
    xy: Vec2,
    size: Vec2,
    color: comfy::Color,
}

struct RenderLine {
    start: Vec2,
    stop: Vec2,
}

struct RenderCircle {
    xy: Vec2,
    size: Vec2,
}

struct RenderText {
    xy: Vec2,
    text: String,
    font_size: f32,
}
struct RenderArrow {
    points: Vec<Vec2>,
}

impl layout::core::format::RenderBackend for RenderStuff {
    fn draw_rect(
        &mut self,
        xy: layout::core::geometry::Point,
        size: layout::core::geometry::Point,
        look: &layout::core::style::StyleAttr,
        _clip: Option<layout::core::format::ClipHandle>,
    ) {
        let mut size = to_mq(size);
        let mut xy = to_mq(xy);
        xy += size * 0.5;
        //size *= 2.0;
        let color = look.line_color.to_web_color();
        let color = u32::from_str_radix(&color[1..], 16).unwrap();
        let color = u32::to_be_bytes(color);
        let color = comfy::Color::rgba8(color[0], color[1], color[2], color[3]);

        self.rects.push(RenderRect { size, xy, color });
        self.update_bounds(xy);
        self.update_bounds(xy + size);
    }

    fn draw_line(
        &mut self,
        start: layout::core::geometry::Point,
        stop: layout::core::geometry::Point,
        _look: &layout::core::style::StyleAttr,
    ) {
        self.lines.push(RenderLine {
            start: to_mq(start),
            stop: to_mq(stop),
        });
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

        self.circles.push(RenderCircle { size, xy });
    }

    fn draw_text(
        &mut self,
        xy: layout::core::geometry::Point,
        text: &str,
        look: &layout::core::style::StyleAttr,
    ) {
        self.texts.push(RenderText {
            xy: to_mq(xy),
            text: text.to_owned(),
            font_size: look.font_size as _,
        });
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
        //let e = path[1].1;
        let mut points = Vec::new();
        points.push(to_mq(s));
        for l in &path[1..] {
            let p = l.1;
            points.push(to_mq(p));
        }
        self.arrows.push(RenderArrow { points });

        //draw_line(to_mq(s), to_mq(e), self.line_size(), BLACK, 0);

        //for l in path[1..].windows(2) {
        //    let s = l[0].1;
        //    let e = l[1].1;
        //    draw_line(to_mq(s), to_mq(e), self.line_size(), BLACK, 0);
        //}
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
    nodes: BTreeMap<u64, String>,
    edges: BTreeMap<EdgeConnection, String>,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct EdgeConnection {
    from: u64,
    to: u64,
}

impl From<&gs_core::Edge> for EdgeConnection {
    fn from(value: &gs_core::Edge) -> Self {
        Self {
            from: value.from,
            to: value.to,
        }
    }
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
                assert!(
                    self.edges.insert((&e).into(), e.label.clone()).is_none(),
                    "Add {:?}",
                    e
                );
            }
            gs_core::Event::RemoveEdge(e) => {
                assert!(self.nodes.contains_key(&e.from), "Remove {:?}", e);
                assert!(self.nodes.contains_key(&e.to), "Remove {:?}", e);
                assert!(
                    self.edges.remove(&((&e).into())).is_some(),
                    "Remove {:?}",
                    e
                )
            }
            gs_core::Event::UpdateEdgeLabel(e, l) => {
                *self.edges.get_mut(&((&e).into())).unwrap() = l;
            }
        }
    }

    fn to_vg(&self) -> Option<VisualGraph> {
        if self.nodes.is_empty() {
            None
        } else {
            let mut out = VisualGraph::new(Orientation::TopToBottom);
            let mut node_map = HashMap::new();
            for n in &self.nodes {
                let sp0 = ShapeKind::new_box(n.1);
                let mut look = StyleAttr::simple();
                let short_name = n.1.as_str().split('0').next().unwrap();
                let col_hex = match short_name {
                    "allocator_ram" => 0xff0000ff,
                    "allocator_vram" => 0xffff00ff,
                    "allocator_vram_raw" => 0xffff00ff,
                    "allocator_vram_image" => 0xffff00ff,
                    "builtin::TransferManager" => 0x00ff00ff,
                    "garbage_collect_ram" => 0x0000ffff,
                    "garbage_collect_vram" => 0x0000ffff,
                    _ => 0x000000ff,
                };

                look.line_color = layout::core::color::Color::new(col_hex);
                //let size = measure_text(n.1, None, 10, 1.0);
                //let sz = Point::new(size.width as _, size.height as _);
                let sz = Point::new(20.0, 5.0);

                let elm = Element::create(sp0, look.clone(), Orientation::LeftToRight, sz);
                let handle = out.add_node(elm);
                node_map.insert(n.0, handle);
            }

            for (e, c) in &self.edges {
                let arrow = Arrow::simple(&c.to_string());
                let from = node_map.get(&e.from);
                let to = node_map.get(&e.to);
                if from.is_none() {
                    panic!("No node for {:?}, {:?}->{:?}", e.from, e.from, e.to);
                }
                if to.is_none() {
                    panic!("No node for {:?}, {:?}->{:?}", e.to, e.from, e.to);
                }
                out.add_edge(arrow, *from.unwrap(), *to.unwrap());
            }

            Some(out)
        }
    }
}

struct GraphTimeline {
    current_graph: Graph,
    current_event_index: usize,
    current_timestep: Timestamp,
    begin_timestep: Timestamp,
    end_timestep: Timestamp,
    events: Vec<(Timestamp, gs_core::Event)>,
    render_elements: RenderStuff,
}

impl GraphTimeline {
    fn new(events: gs_core::EventStream, layout_config: LayoutConfig) -> Self {
        let s = Self {
            current_graph: Default::default(),
            current_event_index: 0,
            current_timestep: events.begin_ts(),
            begin_timestep: events.begin_ts(),
            end_timestep: events.end_ts(),
            events: events.0,
            render_elements: RenderStuff::empty_for_ts(0, layout_config),
        };
        s
    }

    fn get_render(&mut self, layout_config: LayoutConfig) -> (&RenderStuff, bool) {
        let new = if self.current_event_index != self.render_elements.event_index
            || layout_config != self.render_elements.layout_config
        {
            let mut new_render = RenderStuff::empty_for_ts(self.current_event_index, layout_config);

            if let Some(mut vg) = self.current_graph.to_vg() {
                let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    vg.do_it(
                        false,
                        !layout_config.layout_opt,
                        !layout_config.layout_arrange,
                        &mut new_render,
                    )
                }));
                if let Err(err) = res {
                    println!("Error in layout: {:?}", err);
                }
            }
            self.render_elements = new_render;
            true
        } else {
            false
        };

        (&self.render_elements, new)
    }

    fn next(&mut self) {
        if self.current_event_index >= self.events.len() {
            return;
        }

        let (t, e) = &self.events[self.current_event_index];
        self.current_event_index += 1;
        self.current_timestep = *t;
        self.current_graph.apply(e.clone());
    }

    fn prev(&mut self) {
        if self.current_event_index == 0 {
            return;
        }

        self.current_event_index -= 1;
        let (t, e) = &self.events[self.current_event_index];
        self.current_timestep = *t;
        self.current_graph.apply(e.clone().inverse());
    }

    fn go_to_index(&mut self, i: usize) {
        assert!(i <= self.events.len());
        match i.cmp(&self.current_event_index) {
            std::cmp::Ordering::Less => {
                while self.current_event_index != i {
                    self.prev()
                }
            }
            std::cmp::Ordering::Equal => {}
            std::cmp::Ordering::Greater => {
                while self.current_event_index != i {
                    self.next()
                }
            }
        }
    }
    fn go_to_timestep(&mut self, i: Timestamp) {
        assert!(i >= self.begin_timestep);
        assert!(i <= self.end_timestep, "{:?} <= {:?}", i, self.end_timestep);

        match i.cmp(&self.current_timestep) {
            std::cmp::Ordering::Less => {
                while self.current_timestep > i {
                    //println!("Prev: {:?} <= {:?}", self.current_timestep, i);
                    self.prev()
                }
            }
            std::cmp::Ordering::Equal => {}
            std::cmp::Ordering::Greater => {
                while self.current_timestep < i {
                    //println!("Next: {:?} >= {:?}", self.current_timestep, i);
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
    auto_focus: bool,
    layout_config: LayoutConfig,
}

impl MyGame {
    fn make(timeline: GraphTimeline, layout_config: LayoutConfig) -> Self {
        Self {
            timeline,
            prev_mouse: Vec2::new(0.0, 0.0),
            auto_focus: true,
            layout_config,
        }
    }
}

// Everything interesting happens here.
impl GameLoop for MyGame {
    fn new(_c: &mut EngineState) -> Self {
        todo!()
    }

    fn update(&mut self, c: &mut EngineContext) {
        let mut event_index = self.timeline.current_event_index;
        let mut time_step = self.timeline.current_timestep.ms();
        let prev_index = event_index;
        let prev_time = time_step;

        egui().set_style({
            let mut style = egui::Style::default();
            style.spacing.slider_width = c.renderer.width() / 2.0;
            style
        });
        egui::Window::new("Timeline")
            .anchor(egui::Align2::LEFT_BOTTOM, egui::vec2(0.0, 0.0))
            .show(egui(), |ui| {
                ui.add(
                    egui::Slider::new(&mut event_index, 0..=self.timeline.events.len())
                        .text("Time step"),
                );
                ui.add(
                    egui::Slider::new(
                        &mut time_step,
                        self.timeline.begin_timestep.ms()..=self.timeline.end_timestep.ms(),
                    )
                    .text("Time step"),
                );
            });

        if event_index != prev_index {
            self.timeline.go_to_index(event_index);
        }
        if time_step != prev_time {
            self.timeline.go_to_timestep(Timestamp::from_ms(time_step));
        }

        let zoom = {
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
            camera.zoom
        };

        if is_key_pressed(KeyCode::N) {
            self.timeline.next();
        }
        if is_key_pressed(KeyCode::P) {
            self.timeline.prev();
        }
        if is_key_pressed(KeyCode::R) {
            self.timeline.go_to_index(0)
        }
        if is_key_pressed(KeyCode::G) {
            self.timeline.go_to_index(self.timeline.events.len())
        }
        if is_key_pressed(KeyCode::F) {
            self.auto_focus = !self.auto_focus;
        }
        if is_key_pressed(KeyCode::O) {
            self.layout_config.layout_opt = !self.layout_config.layout_opt;
        }
        if is_key_pressed(KeyCode::L) {
            self.layout_config.layout_arrange = !self.layout_config.layout_arrange;
        }

        let (re, new) = self.timeline.get_render(self.layout_config);

        if is_key_pressed(KeyCode::A) || (self.auto_focus && new) {
            if let Some((center, mut size)) = re.region_center_and_size() {
                let mut camera = main_camera_mut();
                camera.center = center;
                size.y *= c.renderer.width() / (c.renderer.height() - 150.0);
                camera.zoom = size.max_element() * 1.05;
            }
        }

        clear_background(WHITE);

        re.draw(zoom);
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

    let layout_config = LayoutConfig {
        layout_arrange: false,
        layout_opt: false,
    };
    let events = gs_core::EventStream::load(&options.input);

    let mut timeline = GraphTimeline::new(events, layout_config);
    println!("Loaded timeline with {} steps", timeline.events.len());
    timeline.go_to_index(100);

    let game = MyGame::make(timeline, layout_config);

    run_comfy_main_async(game, engine).await;
}

fn main() {
    pollster::block_on(run());
}
