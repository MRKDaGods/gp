"use client";

import Image from "next/image";
import { useEffect, useRef, useState } from "react";

export function SplashScreen() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [barWidth, setBarWidth] = useState(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animId = 0;
    const dpr = window.devicePixelRatio || 1;

    const resize = () => {
      canvas.width = window.innerWidth * dpr;
      canvas.height = window.innerHeight * dpr;
      canvas.style.width = `${window.innerWidth}px`;
      canvas.style.height = `${window.innerHeight}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    window.addEventListener("resize", resize);

    const NODES = 60;
    const CONNECT_DIST = 160;
    const nodes = Array.from({ length: NODES }, () => ({
      x: Math.random() * window.innerWidth,
      y: Math.random() * window.innerHeight,
      vx: (Math.random() - 0.5) * 0.4,
      vy: (Math.random() - 0.5) * 0.4,
      r: 1.5 + Math.random() * 1.5,
    }));

    const draw = () => {
      const w = window.innerWidth;
      const h = window.innerHeight;
      ctx.clearRect(0, 0, w, h);

      for (const n of nodes) {
        n.x += n.vx;
        n.y += n.vy;
        if (n.x < 0 || n.x > w) n.vx *= -1;
        if (n.y < 0 || n.y > h) n.vy *= -1;
      }

      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[i].x - nodes[j].x;
          const dy = nodes[i].y - nodes[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < CONNECT_DIST) {
            const alpha = (1 - dist / CONNECT_DIST) * 0.15;
            ctx.strokeStyle = `rgba(96,165,250,${alpha})`;
            ctx.lineWidth = 0.5;
            ctx.beginPath();
            ctx.moveTo(nodes[i].x, nodes[i].y);
            ctx.lineTo(nodes[j].x, nodes[j].y);
            ctx.stroke();
          }
        }
      }

      for (const n of nodes) {
        ctx.fillStyle = "rgba(96,165,250,0.4)";
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
        ctx.fill();
      }

      animId = requestAnimationFrame(draw);
    };
    draw();

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", resize);
    };
  }, []);

  useEffect(() => {
    const t = requestAnimationFrame(() => setBarWidth(100));
    return () => cancelAnimationFrame(t);
  }, []);

  return (
    <div className="fixed inset-0 flex items-center justify-center overflow-hidden bg-[#050a15]">

      {/* Animated network canvas */}
      <canvas ref={canvasRef} className="absolute inset-0" />

      {/* Logo */}
      <div className="relative z-10 flex flex-col items-center">
        <Image
          src="/logo.png"
          alt="ATHAR"
          width={360}
          height={360}
          className="object-contain"
          style={{ filter: "invert(1) brightness(2)" }}
          priority
        />

        {/* Loading bar — driven by React state, no CSS animation flicker */}
        <div className="mt-4 h-[2px] w-48 overflow-hidden rounded-full bg-white/10">
          <div
            className="h-full rounded-full bg-blue-400/60"
            style={{
              width: `${barWidth}%`,
              transition: "width 2.5s ease-out",
            }}
          />
        </div>
      </div>
    </div>
  );
}
