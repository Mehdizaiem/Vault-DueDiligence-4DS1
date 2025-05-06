"use client";

import { useEffect, useRef } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import * as THREE from 'three';
import { 
  FileText, 
  HelpCircle, 
  AlertTriangle,
  TrendingUp,
  BarChart4,
  Newspaper,
  ShieldCheck,
  ArrowRight,
  Zap,
  Lock,
  TrendingDown,
} from "lucide-react";

const features = [
  {
    label: 'Document Analysis',
    icon: FileText,
    color: "text-violet-500",
    bgColor: "bg-violet-50",
    description: "Analyze fund documentation automatically",
    href: "/dashboard/documents"
  },
  {
    label: 'Q&A System',
    icon: HelpCircle,
    color: "text-pink-500",
    bgColor: "bg-pink-50",
    description: "Get instant answers to due diligence queries",
    href: "/dashboard/qa"
  },
  {
    label: 'Risk Analysis',
    icon: AlertTriangle,
    color: "text-orange-500",
    bgColor: "bg-orange-50",
    description: "Monitor real-time risk metrics and alerts",
    href: "/dashboard/risk"
  },
  {
    label: 'Crypto Forecast',
    icon: TrendingUp,
    color: "text-green-500",
    bgColor: "bg-green-50",
    description: "Price predictions and market analysis",
    href: "/dashboard/forecast"
  },
  {
    label: 'Analytics',
    icon: BarChart4,
    color: "text-indigo-500",
    bgColor: "bg-indigo-50",
    description: "Crypto portfolio insights",
    href: "/dashboard/analytics"
  },
  {
    label: 'News',
    icon: Newspaper,
    color: "text-yellow-500",
    bgColor: "bg-yellow-50",
    description: "Latest market events and updates",
    href: "/dashboard/news"
  }
];

// Market pulse data - more valuable than generic stats
const marketPulse = [
  {
    label: 'Bitcoin',
    value: '$67,245',
    change: '+2.4%',
    trend: 'up',
    icon: TrendingUp
  },
  {
    label: 'Market Sentiment',
    value: 'Bullish',
    icon: Zap
  },
  {
    label: 'Risk Rating',
    value: 'Moderate',
    icon: ShieldCheck
  },
  {
    label: 'Ethereum',
    value: '$3,285',
    change: '-1.2%',
    trend: 'down',
    icon: TrendingDown
  }
];

export default function DashboardPage() {
  const canvasRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!canvasRef.current) return;
    
    let scene: THREE.Scene;
    let camera: THREE.PerspectiveCamera;
    let renderer: THREE.WebGLRenderer;
    const blocks: {
      mesh?: THREE.Mesh;
      initialPosition: THREE.Vector3;
      speed: number;
      angle: number;
      radius?: number;
    }[] = [];
    let animationFrameId: number;
    
    // Initialize Three.js
    const initThree = () => {
      // Scene setup
      scene = new THREE.Scene();
      camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
      
      renderer = new THREE.WebGLRenderer({ 
        antialias: true, 
        alpha: true,
        canvas: document.createElement('canvas')
      });
      renderer.setSize(window.innerWidth, window.innerHeight);
      renderer.setClearColor(0xffffff, 0); // Transparent background
      renderer.setPixelRatio(window.devicePixelRatio);
      
      // Add renderer to the DOM
      if (canvasRef.current) {
        canvasRef.current.innerHTML = '';
        canvasRef.current.appendChild(renderer.domElement);
      }
      
      // Add subtle lighting
      const ambientLight = new THREE.AmbientLight(0x404040, 1.5);
      scene.add(ambientLight);
      
      const directionalLight = new THREE.DirectionalLight(0x4c6bff, 0.8);
      directionalLight.position.set(0, 1, 1);
      scene.add(directionalLight);
      
      const pointLight = new THREE.PointLight(0x4c6bff, 0.8, 100);
      pointLight.position.set(10, 10, 10);
      scene.add(pointLight);
      
      // Create blockchain visualization
      createBlockchain();
      
      // Position camera
      camera.position.z = 20;
      
      // Set up event listeners
      window.addEventListener('resize', onWindowResize);
      
      // Start animation loop
      animate();
    };
    
    // Create blockchain blocks
    const createBlockchain = () => {
      const blockGeometry = new THREE.BoxGeometry(2, 1, 0.5);
      const blockMaterial = new THREE.MeshPhongMaterial({
        color: 0x4c6bff,
        specular: 0x4c6bff,
        shininess: 30,
        transparent: true,
        opacity: 0.5,
      });
      
      const totalBlocks = 12;
      for (let i = 0; i < totalBlocks; i++) {
        const block = new THREE.Mesh(blockGeometry, blockMaterial.clone());
        
        // Position in a simple circle pattern
        const angle = i * (Math.PI * 2 / totalBlocks);
        const radius = 15;
        
        block.position.x = Math.cos(angle) * radius;
        block.position.y = Math.sin(angle) * radius;
        block.position.z = -5 + Math.random() * 5;
        
        // Random rotation
        block.rotation.x = Math.random() * 0.3 - 0.15;
        block.rotation.y = Math.random() * 0.3 - 0.15;
        block.rotation.z = Math.random() * 0.3 - 0.15;
        
        // Store block data for animation
        blocks.push({
          mesh: block,
          initialPosition: block.position.clone(),
          speed: 0.005 + Math.random() * 0.003,
          angle: angle,
          radius: radius
        });
        
        scene.add(block);
      }
    };
    
    // Update blockchain position and rotation
    const updateBlockchain = (time: number) => {
      // Update blocks
      blocks.forEach((block, index) => {
        if (block.mesh && block.radius !== undefined) {
          // Only animate if it's a regular block with a mesh
          const offset = index * 0.5;
          
          // Update block position in circular pattern
          block.angle += block.speed;
          const newRadius = block.radius + Math.sin(time * 0.0005 + offset) * 0.5;
          
          block.mesh.position.x = Math.cos(block.angle) * newRadius;
          block.mesh.position.y = Math.sin(block.angle) * newRadius;
          
          // Subtle rotation
          block.mesh.rotation.x += 0.003;
          block.mesh.rotation.y += 0.002;
        }
      });
    };
    
    // Animation loop
    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      
      // Rotate entire scene slowly for subtle effect
      if (scene) {
        scene.rotation.y += 0.0005;
        scene.rotation.x = Math.sin(Date.now() * 0.0001) * 0.1;
        
        // Update blockchain
        updateBlockchain(Date.now());
        
        // Render scene
        renderer.render(scene, camera);
      }
    };
    
    // Window resize handler
    const onWindowResize = () => {
      if (camera && renderer) {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
      }
    };
    
    // Initialize everything
    initThree();
    
    // Cleanup on unmount
    return () => {
      window.removeEventListener('resize', onWindowResize);
      cancelAnimationFrame(animationFrameId);
      
      // Dispose of resources
      if (renderer) {
        renderer.dispose();
      }
      
      if (scene) {
        scene.clear();
      }
    };
  }, []);

  return (
    <div className="relative min-h-screen flex flex-col overflow-hidden">
      {/* 3D Background Canvas */}
      <div 
        ref={canvasRef} 
        className="fixed inset-0 pointer-events-none z-0 opacity-60"
      />
      
      {/* Content */}
      <div className="relative z-10 flex-1 flex flex-col">
        <div className="max-w-7xl mx-auto w-full px-8 py-6">
        {/* Welcome Banner */}
        <div className="flex flex-col items-center mb-8">
          <div className="w-full p-6 rounded-2xl bg-gradient-to-br from-[#1a1a3a] to-[#0f0f2a] text-white shadow-lg">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="flex items-center">
                  <h1 className="text-3xl font-bold text-white"
                    style={{
                      textShadow: '0 0 10px rgba(76, 107, 255, 0.8), 0 0 20px rgba(76, 107, 255, 0.4)'
                    }}
                  >
                    VAULT
                  </h1>
                </div>
                <div>
                  <p className="text-lg text-gray-200">Cryptocurrency intelligence platform for fund monitoring and compliance</p>
                </div>
              </div>
              <div className="hidden md:block">
                <div className="flex gap-4">
                  <div className="p-3 rounded-lg bg-white/10 backdrop-blur-sm">
                    <Lock className="h-6 w-6 text-white" />
                  </div>
                  <div className="p-3 rounded-lg bg-white/10 backdrop-blur-sm">
                    <ShieldCheck className="h-6 w-6 text-white" />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

          {/* Market Pulse - More valuable than generic stats */}
          <div className="mb-8">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Market Pulse</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {marketPulse.map((item, index) => (
                <div 
                  key={item.label} 
                  className="bg-white/70 backdrop-blur-sm rounded-xl border border-gray-100 p-4 shadow-sm hover:shadow-md transition-shadow"
                >
                  <div className="flex items-center gap-3">
                    <div className={`p-3 rounded-lg ${
                      index === 0 ? 'bg-green-50' : 
                      index === 1 ? 'bg-blue-50' : 
                      index === 2 ? 'bg-purple-50' : 
                      'bg-red-50'
                    }`}>
                      <item.icon className={`h-5 w-5 ${
                        index === 0 ? 'text-green-500' : 
                        index === 1 ? 'text-blue-500' : 
                        index === 2 ? 'text-purple-500' : 
                        'text-red-500'
                      }`} />
                    </div>
                    <div>
                      <div className="text-sm text-gray-500">{item.label}</div>
                      <div className="flex items-center gap-2">
                        <span className="text-xl font-semibold text-gray-800">
                          {item.value}
                        </span>
                        {item.change && (
                          <span className={`text-xs font-medium ${
                            item.trend === 'up' ? 'text-green-500' : 'text-red-500'
                          }`}>
                            {item.change}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Features Grid - Minimalist, Professional */}
          <h2 className="text-xl font-semibold text-gray-800 mb-4">Platform Tools</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {features.map((feature) => (
              <Link
                key={feature.label}
                href={feature.href}
                className="group"
              >
                <div className="bg-white/70 backdrop-blur-sm rounded-xl border border-gray-100 p-5 h-full hover:shadow-md transition-all flex flex-col">
                  <div className="flex items-start gap-4 mb-3">
                    <div className={`${feature.bgColor} p-3 rounded-xl`}>
                      <feature.icon className={`h-5 w-5 ${feature.color}`} />
                    </div>
                    <div>
                      <h3 className="text-lg font-medium text-gray-800 mb-1">{feature.label}</h3>
                      <p className="text-gray-500 text-sm">{feature.description}</p>
                    </div>
                  </div>
                  <div className="mt-auto pt-3 flex justify-end">
                    <div className="text-sm font-medium text-blue-600 group-hover:text-blue-700 flex items-center gap-1">
                      <span>Open</span>
                      <ArrowRight size={16} className="transition-transform group-hover:translate-x-1" />
                    </div>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}