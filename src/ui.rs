// src/ui.rs

use crate::adaptive::{MinerMetrics, UiMessage};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{Duration, Instant};
use metal::Buffer;

// ✅ Fix constants import
use crate::constants::{LANES, NONCES_PER_THREAD};
use crate::sha_helpers::{aligned_u32_buffer, aligned_f32_buffer};

use crossterm::{
    event::{self, Event as CEvent, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use tui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    symbols,
    text::{Span, Spans},
    widgets::{Block, Borders, Gauge, Paragraph, Wrap, Chart, Axis, Dataset},
    Terminal,
};

pub async fn run_ui(
    metrics: Arc<RwLock<MinerMetrics>>,
    mut ui_rx: tokio::sync::mpsc::UnboundedReceiver<UiMessage>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut logs: Vec<String> = Vec::new();

    loop {
        while let Ok(msg) = ui_rx.try_recv() {
            match msg {
                UiMessage::Metrics(updated) => {
                    let mut m = metrics.write().await;
                    *m = updated;
                }
                UiMessage::Status(line) => {
                    logs.push(line);
                    if logs.len() > 12 {
                        logs.remove(0);
                    }
                }
            }
        }

        let m = metrics.read().await.clone();

        terminal.draw(|f| {
            let size = f.size();
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .margin(1)
                .constraints(
                    [
                        Constraint::Length(3),
                        Constraint::Length(5),
                        Constraint::Length(7),
                        Constraint::Min(4),
                        Constraint::Length(6),
                    ]
                    .as_ref(),
                )
                .split(size);

            let header = Paragraph::new(format!(
                "🧠 Rust Metal Miner — {:.3} MH/s | Total Hashes: {:>12}",
                m.hashrate_mhs, m.total_hashes
            ))
            .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
            .block(Block::default().borders(Borders::ALL).title("Status"));
            f.render_widget(header, chunks[0]);

            let gauge_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints(
                    [
                        Constraint::Percentage(33),
                        Constraint::Percentage(33),
                        Constraint::Percentage(34),
                    ]
                    .as_ref(),
                )
                .split(chunks[1]);

            let mask_g = Gauge::default()
                .block(Block::default().borders(Borders::ALL).title("Mask"))
                .ratio(m.mask.clamp(0.0, 1.0) as f64)
                .style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
                .label(Span::raw(format!("{:.3}", m.mask)));
            f.render_widget(mask_g, gauge_chunks[0]);

            let prune_g = Gauge::default()
                .block(Block::default().borders(Borders::ALL).title("Prune"))
                .ratio(m.prune.clamp(0.0, 1.0) as f64)
                .style(Style::default().fg(Color::Green))
                .label(Span::raw(format!("{:.3}", m.prune)));
            f.render_widget(prune_g, gauge_chunks[1]);

            let gain_g = Gauge::default()
                .block(Block::default().borders(Borders::ALL).title("Gain"))
                .ratio(m.gain.clamp(0.0, 1.0) as f64)
                .style(Style::default().fg(Color::Magenta))
                .label(Span::raw(format!("{:.3}", m.gain)));
            f.render_widget(gain_g, gauge_chunks[2]);

            let lane_lines: Vec<Spans> = if !m.avg_post.is_empty() {
                itertools::izip!(&m.avg_post, &m.avg_fwht, &m.avg_cs)
                    .enumerate()
                    .map(|(i, (p, fwh, c))| {
                        Spans::from(vec![
                            Span::styled(
                                format!("Lane {:>2}: ", i),
                                Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD),
                            ),
                            Span::raw(format!("POST={:.3}  FWHT={:.3}  CS={:.3}", p, fwh, c)),
                        ])
                    })
                    .collect()
            } else {
                vec![Spans::from("No telemetry yet…")]
            };
            let lane_panel = Paragraph::new(lane_lines)
                .block(Block::default().borders(Borders::ALL).title("Lane Metrics"));
            f.render_widget(lane_panel, chunks[2]);

            let nibble_lines: Vec<Spans> = m
                .nibble_tree
                .iter()
                .enumerate()
                .map(|(i, row)| {
                    Spans::from(Span::styled(
                        format!("Row {} → {:?}", i, row),
                        Style::default().fg(Color::Gray),
                    ))
                })
                .collect();
            let nibble_panel = Paragraph::new(nibble_lines)
                .block(Block::default().borders(Borders::ALL).title("Nibble Tree & Diagnostics"))
                .wrap(Wrap { trim: true });
            f.render_widget(nibble_panel, chunks[3]);

            let log_lines: Vec<Spans> = logs
                .iter()
                .map(|l| Spans::from(Span::styled(l, Style::default().fg(Color::LightCyan))))
                .collect();
            let log_panel = Paragraph::new(log_lines)
                .block(Block::default().borders(Borders::ALL).title("Miner Log"))
                .wrap(Wrap { trim: true });
            f.render_widget(log_panel, chunks[4]);
        })?;

        if event::poll(Duration::from_millis(100))? {
            if let CEvent::Key(key) = event::read()? {
                if matches!(key.code, KeyCode::Char('q') | KeyCode::Esc) {
                    disable_raw_mode()?;
                    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
                    terminal.show_cursor()?;
                    break;
                }
            }
        }

        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    Ok(())
}
