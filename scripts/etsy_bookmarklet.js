// Etsy 媒体 URL 提取书签脚本
//
// 使用方法：
// 1. 在浏览器中创建新书签
// 2. 将下面的代码复制到书签的 URL 字段
// 3. 打开 Etsy 商品页面，点击书签即可提取 URL
//
// 书签代码（复制这一行）：
// javascript:(function(){var imgs=[],vids=[];document.querySelectorAll('img[src*="etsystatic"]').forEach(i=>{var s=i.src.replace(/il_\d+x\d+/,'il_1588xN').replace(/il_\d+xN/,'il_1588xN');if(s.includes('il_')&&!imgs.includes(s))imgs.push(s)});document.querySelectorAll('video source, video[src]').forEach(v=>{var s=v.src||v.getAttribute('src');if(s&&s.includes('etsystatic')&&!vids.includes(s))vids.push(s)});var r='IMAGES:\n'+imgs.join('\n')+'\n\nVIDEOS:\n'+vids.join('\n');console.log(r);prompt('复制以下URL (Cmd+A, Cmd+C):',r);})();

// 完整版（带格式化输出）：
(function() {
    var images = [];
    var videos = [];

    // 提取图片
    document.querySelectorAll('img[src*="etsystatic"]').forEach(function(img) {
        var src = img.src;
        // 转换为高清版本
        src = src.replace(/il_\d+x\d+/, 'il_1588xN');
        src = src.replace(/il_\d+xN/, 'il_1588xN');
        if (src.includes('il_') && !images.includes(src)) {
            images.push(src);
        }
    });

    // 提取视频
    document.querySelectorAll('video source, video[src]').forEach(function(video) {
        var src = video.src || video.getAttribute('src');
        if (src && src.includes('etsystatic') && !videos.includes(src)) {
            videos.push(src);
        }
    });

    // 尝试从页面数据中提取
    var scripts = document.querySelectorAll('script');
    scripts.forEach(function(script) {
        var content = script.innerHTML;

        // 视频 URL
        var videoMatches = content.match(/https:\/\/v\.etsystatic\.com\/video\/upload\/[^"'\s]+\.mp4/g);
        if (videoMatches) {
            videoMatches.forEach(function(url) {
                if (!videos.includes(url)) videos.push(url);
            });
        }
    });

    // 输出结果
    var result = {
        listing_id: window.location.pathname.match(/\/listing\/(\d+)/)?.[1] || 'unknown',
        title: document.querySelector('h1')?.innerText || document.title,
        url: window.location.href,
        images: images,
        videos: videos
    };

    console.log('=== Etsy Media URLs ===');
    console.log(JSON.stringify(result, null, 2));

    // 复制到剪贴板
    var text = 'IMAGES:\n' + images.join('\n') + '\n\nVIDEOS:\n' + videos.join('\n');

    if (navigator.clipboard) {
        navigator.clipboard.writeText(JSON.stringify(result, null, 2)).then(function() {
            alert('已复制 ' + images.length + ' 个图片和 ' + videos.length + ' 个视频 URL 到剪贴板！\n\n请在终端运行:\npython download_urls.py');
        });
    } else {
        prompt('复制以下内容:', text);
    }
})();
