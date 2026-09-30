# **3D Floor Plan to 2D Room Layout — Take Home Assignment for CV/ML Engineer at COXIT**

## **Objective**

Given one 3D apartment floor plan image, build a 4–6-hour exploratory prototype that approximates a 2D room layout using classical computer vision or simple segmentation techniques.

The goal is to create a transparent pipeline that identifies visible room-like regions, approximates their boundaries, and estimates relative pixel-based areas.

## **Requirements**

- Identify visible room-like regions.
- Approximate room boundaries.
- Estimate relative pixel-based areas.
- Clearly document all assumptions, such as:
    - Top-down view.
    - Distinguishable floor/wall colors.
    - Limited occlusion.
- Manual preprocessing or light manual correction is acceptable if documented.
- The solution should be submitted as a GitHub repository.
- The code should be containerized and runnable via Docker.
- Both Dockerfile and docker-compose.yml should be provided for easy setup.

## **Deliverables**

1. **Annotated Image**
    - An image showing the detected/approximated room regions and boundaries.
2. **JSON Output**
    - A JSON file containing room polygons and relative areas.
3. **Source Code**
    - Dockerfile included.
    - docker-compose.yml included.
4. **README.md**
    - Short description of the approach.
    - Assumptions.
    - Limitations.
    - Next steps.

