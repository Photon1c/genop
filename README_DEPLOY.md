# Deployment Guide for Generative Optogenetics Dashboard

## Quick Deploy to Netlify

### Option 1: Static Site (Recommended for Netlify)

The dashboard is a standalone HTML file that works without a server:

1. **Via Netlify Drop:**
   - Go to https://app.netlify.com/drop
   - Drag and drop the `genop` folder
   - Your site will be live instantly!

2. **Via Git:**
   - Push your code to GitHub/GitLab
   - Connect repository to Netlify
   - Netlify will auto-detect and deploy

3. **Via Netlify CLI:**
   ```bash
   npm install -g netlify-cli
   cd genop
   netlify deploy --prod
   ```

### Option 2: Flask App (For Local Development or Other Hosting)

For local development or hosting on platforms that support Python (Heroku, Railway, Render):

```bash
cd genop
pip install -r requirements.txt
python app.py
```

Then visit `http://localhost:5000`

## Features

✅ **Fully Standalone**: `dashboard.html` works without any server  
✅ **Desensitization**: Already implemented in JavaScript  
✅ **Interactive Controls**: Sliders for all parameters  
✅ **Multiple Controllers**: Greedy, Random, Pulse  
✅ **Real-time Visualization**: Three.js-based 3D rendering  

## File Structure

```
genop/
├── dashboard.html      # Main dashboard (standalone, works on Netlify)
├── app.py              # Flask app (for local dev or Python hosting)
├── requirements.txt    # Python dependencies
├── netlify.toml        # Netlify configuration
├── static.json         # Alternative static config
├── test_script.py      # Base environment
├── test_script2.py     # Advanced features + CMA-ES
└── README.md           # Main documentation
```

## Notes

- **Netlify**: Best for static sites. The dashboard works perfectly as-is.
- **Flask**: Use `app.py` for local development or if you want to add Python-based API endpoints.
- **CMA-ES**: Currently implemented in Python (`test_script2.py`). For browser-based optimization, consider using a JavaScript CMA-ES library.

## Future Enhancements

- Add Netlify Functions for serverless Python optimization
- Integrate JavaScript CMA-ES library for browser-based optimization
- Add data export/import features
- Add parameter preset saving/loading

