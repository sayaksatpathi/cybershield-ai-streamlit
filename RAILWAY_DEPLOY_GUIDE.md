# 🚂 Railway Backend Deployment Guide

## 🚀 **Deploy Your Flask Backend to Railway**

Your backend files are ready for Railway deployment!

### 📦 **Ready Files:**
- ✅ `simple_backend.py` - Production Flask API
- ✅ `Procfile` - Railway startup configuration  
- ✅ `requirements.txt` - Python dependencies
- ✅ GitHub repository with all code

---

## 🚂 **Railway Deployment Steps**

### **Step 1: Visit Railway**
```
🌐 URL: https://railway.app
```

### **Step 2: Sign Up / Sign In**
- Click "Login" or "Start a New Project"
- Sign in with GitHub account
- Authorize Railway to access your repositories

### **Step 3: Deploy from GitHub**
- Click "Deploy from GitHub repo"
- Select: `sayaksatpathi/cybershield-ai-streamlit`
- Railway will automatically detect your Python project

### **Step 4: Auto-Configuration**
Railway will automatically:
- ✅ Detect `Procfile` (`web: python simple_backend.py`)
- ✅ Install dependencies from `requirements.txt`
- ✅ Set up Python environment
- ✅ Deploy your Flask API

### **Step 5: Get Your Backend URL**
After deployment (2-3 minutes), you'll get:
```
https://your-app-name.railway.app
```

**Example**: `https://cybershield-ai-backend-production.railway.app`

---

## 🔧 **Environment Variables (Optional)**

Set these in Railway dashboard if needed:
```
FLASK_ENV=production
PORT=5000
CORS_ORIGINS=*
```

---

## 🧪 **Test Your Deployed Backend**

Once deployed, test with:
```bash
curl https://your-app.railway.app/api/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "version": "2.1.0",
  "features": {
    "models_trained": false,
    "available_models": ["Random Forest"],
    "max_upload_size": "1GB"
  }
}
```

---

## ⚡ **Quick Deploy Button**

Railway provides one-click deployment. After you connect your GitHub account:

1. **Visit**: https://railway.app/new
2. **Select**: "Deploy from GitHub repo"
3. **Choose**: `sayaksatpathi/cybershield-ai-streamlit`
4. **Deploy**: Automatic detection and deployment

**Deployment time**: ~2-3 minutes
**Cost**: Free tier available (500 hours/month)

---

## 🎯 **What Happens Next**

After Railway deployment:
1. ✅ Your Flask API will be live on Railway
2. 🔧 Update frontend API URLs to point to Railway
3. 📱 Re-deploy frontend to Netlify
4. 🎉 Full-stack application complete!

**Ready to deploy? Visit https://railway.app and deploy your backend!** 🚂
