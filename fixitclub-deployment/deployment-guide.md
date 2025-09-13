# Free Deployment Guide for FixItClub API

## Option 1: Railway (Recommended)

1. **Export your code**: Download all files from this Replit
2. **Create GitHub repo**: Upload your code
3. **Sign up at Railway.app** (free)
4. **Deploy**: Connect your GitHub repo
5. **Set run command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

## Option 2: Render

1. **Create account** at render.com
2. **Connect GitHub** repo with your code
3. **Choose "Web Service"**
4. **Set build command**: `pip install -r requirements.txt`
5. **Set start command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

## Option 3: Vercel (Serverless)

1. **Install Vercel CLI**: `npm i -g vercel`
2. **Create vercel.json** in root:
```json
{
  "builds": [{"src": "api/main.py", "use": "@vercel/python"}],
  "routes": [{"src": "/(.*)", "dest": "api/main.py"}]
}
```
3. **Deploy**: `vercel --prod`

## Required Files to Export:
- `api/` folder (all files)
- `models/` folder
- `utils/` folder  
- `examples/` folder
- `app.py` (if keeping Streamlit)
- All Python files

## Environment Variables Needed:
- No special secrets required
- Your API will work out of the box!

## Expected Result:
Your API will be available at URLs like:
- Railway: `https://your-app.railway.app/fixitclub/docs`
- Render: `https://your-app.onrender.com/fixitclub/docs`
- Vercel: `https://your-app.vercel.app/fixitclub/docs`

## Pro Tip:
Railway is the easiest - just connect GitHub and it auto-deploys!