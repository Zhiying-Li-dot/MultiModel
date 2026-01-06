// PVTT Data Collector - Content Script
// 在 Etsy 商品页面中运行，提取视频和图片

function extractMediaFromPage() {
  const data = {
    listingId: null,
    title: '',
    url: window.location.origin + window.location.pathname,  // 简洁URL，不含查询参数
    videos: [],
    images: [],
    breadcrumbs: [],  // Etsy 原始分类路径
    timestamp: new Date().toISOString()
  };

  // 提取 listing ID 和 product handle
  const match = window.location.pathname.match(/\/listing\/(\d+)\/([^\/]+)/);
  if (match) {
    data.listingId = match[1];
    data.productHandle = match[2];
  }

  // 提取标题
  const titleEl = document.querySelector('h1');
  if (titleEl) {
    data.title = titleEl.innerText.trim();
  }

  // 提取面包屑导航（完整记录 Etsy 所有层级，不做任何修改）
  const breadcrumbLinks = document.querySelectorAll('nav[aria-label="breadcrumb"] a, nav ol a, [data-ui="list-item-breadcrumbs"] a');
  breadcrumbLinks.forEach(link => {
    const text = link.innerText.trim();
    if (text && text !== 'Homepage') {
      data.breadcrumbs.push(text);
    }
  });

  // 提取视频：从页面 HTML 提取 contentURL
  const pageContent = document.documentElement.innerHTML;
  const contentUrlMatches = pageContent.matchAll(/"contentURL"\s*:\s*"([^"]+)"/g);
  for (const match of contentUrlMatches) {
    const url = match[1].replace(/\\\//g, '/');
    if (url.includes('v.etsystatic.com')) {
      let videoUrl = url;
      if (!videoUrl.endsWith('.mp4')) {
        videoUrl += '.mp4';
      }
      if (!data.videos.includes(videoUrl)) {
        data.videos.push(videoUrl);
      }
    }
  }

  // 提取图片：只从商品图片轮播区域提取（排除推荐商品）
  // 查找 data-src-zoom-image 属性（这些是当前商品的高清图片）
  const carouselImages = document.querySelectorAll('[data-src-zoom-image]');
  carouselImages.forEach(img => {
    const zoomUrl = img.getAttribute('data-src-zoom-image');
    if (zoomUrl && zoomUrl.includes('etsystatic.com') && zoomUrl.includes('il_fullxfull')) {
      if (!data.images.includes(zoomUrl)) {
        data.images.push(zoomUrl);
      }
    }
  });

  return data;
}

// 监听来自 popup 的请求
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log('Content script received message:', request);

  if (request.action === 'extractMedia') {
    try {
      const data = extractMediaFromPage();
      console.log('Extracted data:', data);
      sendResponse(data);
    } catch (error) {
      console.error('Error extracting data:', error);
      sendResponse({ error: error.message });
    }
  }
  return true; // 保持消息通道开启，支持异步响应
});

console.log('PVTT Data Collector: Content script loaded on', window.location.href);
