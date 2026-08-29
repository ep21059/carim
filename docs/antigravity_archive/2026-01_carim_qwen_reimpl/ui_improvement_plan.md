# Viewer UI Improvement Proposal

To elevate the CARIM Viewer from a debugging tool to a premium demonstration interface, I propose the following design enhancements.

## 1. Design Philosophy: "Dark & Dynamic"
The current default Streamlit look is functional but generic. We will apply a custom CSS injection to create a "Dashboard" feel.
-   **Theme**: Dark Mode base with Neon Accents (Cyan/Magenta) to match the "Future/AI" vibe of Autonomous Driving.
-   **Typography**: Use monospaced fonts for data/scores, and clean sans-serif (Inter/Roboto) for descriptions.

## 2. Layout & Components

### A. The "Result Card" (Major Change)
Instead of a simple list, each search result will be a distinct "Card" container with a glass-morphism effect.

**Structure:**
-   **Header**: `Scene ID` (Mono) | `Score` (Badge with Color Gradient) | `Weather` (Icon).
-   **Body (Split 60/40)**:
    -   **Left**: Video Player (Larger, centered).
    -   **Right**:
        -   **Matches**: "Why this scene?" (Top 3 matching elements as pills suitable for quick scanning).
        -   **Context**: Location, Time (Minimalist icons).
-   **Footer (Expandable)**: "Detailed Analysis" (Click to slide down the Token Heatmap & Full Element List).

### B. Enhanced Video Player
-   **Timeline Slider**: Replace the basic numeric slider with a "Progress Bar" visual.
-   **Highlight Markers**: Visually mark the exact frame where the event occurs on the timeline (e.g., a yellow dot on the progress bar).

### C. Sidebar "Control Center"
-   Group filters into collapsible sections.
-   Add a "Dataset Stats" widget at the top (Total Scenes, Index Size).

## 3. Explainability Visuals (The "Wow" Factor)

### Current
-   Red/Gray boxes for token scores. Functional but cluttered.

### Proposed
-   **Interactive Highlighting**:
    -   Display the Query Sentence.
    -   When hovering over a "High Attention" token (e.g., "Pedestrian"), highlight the corresponding Element in the list.
    -   (Note: Hover interaction requires custom JS/Streamlit Components, fallback is static coloring).
-   **Score Decomposition**:
    -   A small horizontal bar chart showing contribution of top 5 elements to the total score.

## 4. Implementation Details

### Custom CSS (`assets/style.css`)
```css
/* Example Glassmorphism Card */
.stContainer {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 20px;
    backdrop-filter: blur(10px);
}
```

### Python Component Update
-   **`render_result_card(scene, score, details)`**: A new helper function to encapsulate this layout logic.
-   **`plot_score_breakdown(elements, scores)`**: Function to generate a Plotly/Altair chart for the score breakdown (richer than HTML text).

## 5. Execution Plan
1.  **Refactor**: Separate `app.py` into `ui_components.py` (Layouts) and `app.py` (Logic).
2.  **Style**: Create `style.css` and inject it.
3.  **Visualization**: Replace raw HTML heatmaps with `st.altair_chart` for clearer score distribution.
