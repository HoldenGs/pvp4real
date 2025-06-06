# MetaUrban × PVP4Real Integration Website

This directory contains the GitHub Pages website showcasing the integration of MetaUrban's sidewalk navigation environment with PVP4Real's human-in-the-loop reinforcement learning framework.

## Features

- **Modern Design**: Responsive, visually engaging layout with smooth animations
- **Interactive Elements**: Code explorer with syntax highlighting, video player, animated metrics
- **Technical Documentation**: Detailed explanation of the integration work
- **Demo Video**: Embedded demonstration of the system in action
- **Architecture Visualization**: Clear system architecture diagram

## Setup Instructions

### 1. Enable GitHub Pages

1. Go to your repository settings
2. Navigate to "Pages" section
3. Set source to "Deploy from a branch"
4. Select branch: `main` (or your default branch)
5. Select folder: `/docs`
6. Click "Save"

### 2. Add Your Demo Video

Replace the video source in `index.html`:
```html
<source src="path/to/your/demo3.mp4" type="video/mp4">
```

### 3. Customize Content

Edit `index.html` to:
- Update repository links
- Modify contact information
- Adjust technical details
- Add any additional sections

## File Structure

```
docs/
├── index.html          # Main website file
├── styles.css          # All styling and animations
├── script.js           # Interactive functionality
├── _config.yml         # Jekyll configuration
└── README.md          # This file
```

## Key Sections

1. **Hero Section**: Eye-catching introduction with animated gradient text
2. **Problem & Solution**: Explains the challenges and your approach
3. **Technical Contribution**: Four key aspects of your work
4. **Code Explorer**: Interactive tabs showing your code
5. **Demo Video**: Embedded video demonstration
6. **Architecture**: System component diagram
7. **Results**: Impact and achievements
8. **Footer**: Links and additional information

## Customization

### Colors
The site uses a modern color palette:
- Primary: `#ff6b6b` (coral red)
- Secondary: `#4ecdc4` (teal)
- Accent: `#45b7d1` (blue)
- Dark: `#2c3e50` (navy)

### Animations
- Gradient text animation
- Scroll-triggered element animations
- Hover effects on cards and buttons
- Parallax scrolling on hero section

### Interactive Features
- Tabbed code explorer
- Video player with custom overlay
- Copy-to-clipboard for code blocks
- Smooth scrolling navigation
- Animated counters for metrics

## Browser Support

The website is designed to work on:
- Modern browsers (Chrome, Firefox, Safari, Edge)
- Mobile devices (responsive design)
- Tablets and desktops

## Performance

The site is optimized for:
- Fast loading with minimal dependencies
- Smooth animations (60fps)
- Lazy loading for images/videos
- Efficient CSS and JavaScript

Your website will be available at: `https://yourusername.github.io/repository-name/`

## Tips for Best Results

1. **Video Optimization**: Compress your demo video for web (H.264, reasonable bitrate)
2. **GitHub Repository**: Update the GitHub links to point to your actual repository
3. **Contact Information**: Add your actual contact details in the footer
4. **SEO**: Update the meta tags and title for better search engine visibility
5. **Analytics**: Consider adding Google Analytics if you want to track visitors

The website is fully self-contained and doesn't require any build process - GitHub Pages will serve it directly! 