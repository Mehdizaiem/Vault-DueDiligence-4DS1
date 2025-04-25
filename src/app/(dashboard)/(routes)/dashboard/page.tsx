"use client";

import { useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import * as THREE from 'three';
import { Card } from "@/components/ui/card";
import { 
  FileText, 
  HelpCircle, 
  AlertTriangle,
  Users,
  DollarSign,
  Shield,
  Activity,
  TrendingUp,
  BarChart4,
  Newspaper
} from "lucide-react";

const stats = [
  {
    label: 'Total Funds',
    value: '28',
    icon: Users,
    color: 'text-blue-500',
    bgColor: 'bg-blue-100'
  },
  {
    label: 'AUM',
    value: '$2.4B',
    icon: DollarSign,
    color: 'text-green-500',
    bgColor: 'bg-green-100'
  },
  {
    label: 'Risk Score',
    value: '94%',
    icon: Shield,
    color: 'text-violet-500',
    bgColor: 'bg-violet-100'
  },
  {
    label: 'Active Alerts',
    value: '7',
    icon: Activity,
    color: 'text-pink-500',
    bgColor: 'bg-pink-100'
  }
];

const features = [
  {
    label: 'Document Analysis',
    icon: FileText,
    color: "text-violet-500",
    bgColor: "bg-violet-100",
    description: "Upload and analyze fund documentation automatically",
    href: "/dashboard/documents"
  },
  {
    label: 'Q&A System',
    icon: HelpCircle,
    color: "text-pink-500",
    bgColor: "bg-pink-100",
    description: "Get instant answers to due diligence queries",
    href: "/dashboard/qa"
  },
  {
    label: 'Risk Analysis',
    icon: AlertTriangle,
    color: "text-orange-500",
    bgColor: "bg-orange-100",
    description: "Monitor real-time risk metrics and alerts",
    href: "/dashboard/risk"
  },
  {
    label: 'Crypto Forecast',
    icon: TrendingUp,
    color: "text-green-500",
    bgColor: "bg-green-100",
    description: "Price predictions and market analysis for cryptocurrencies",
    href: "/dashboard/forecast"
  },
  {
    label: 'Analytics',
    icon: BarChart4,
    color: "text-indigo-500",
    bgColor: "bg-indigo-100",
    description: "Comprehensive analytics and insights for crypto portfolios",
    href: "/dashboard/analytics"
  },
  {
    label: 'News',
    icon: Newspaper,
    color: "text-yellow-500",
    bgColor: "bg-yellow-100",
    description: "Latest news, regulatory updates, and market events",
    href: "/dashboard/news"
  }
];

interface BlockData {
  mesh?: THREE.Mesh;
  initialPosition: THREE.Vector3;
  speed: number;
  angle: number;
  radius?: number;
  verticalOffset?: number;
  particleSystem?: THREE.Points;
  positions?: Float32Array;
  initialPositions?: Float32Array;
}

interface ConnectionData {
  line: THREE.Line;
  startBlock: BlockData;
  endBlock: BlockData;
}

export default function DashboardPage() {
  const canvasRef = useRef<HTMLDivElement>(null);
  const [hoveredStat, setHoveredStat] = useState<number | null>(null);
  const [hoveredFeature, setHoveredFeature] = useState<number | null>(null);

  useEffect(() => {
    if (!canvasRef.current) return;
    
    let scene: THREE.Scene;
    let camera: THREE.PerspectiveCamera;
    let renderer: THREE.WebGLRenderer;
    const blocks: BlockData[] = [];
    const connections: ConnectionData[] = [];
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
    
    // Create blockchain blocks and connections
    const createBlockchain = () => {
      const blockGeometry = new THREE.BoxGeometry(2, 1, 0.5);
      const blockMaterial = new THREE.MeshPhongMaterial({
        color: 0x4c6bff,
        specular: 0x4c6bff,
        shininess: 30,
        transparent: true,
        opacity: 0.5,
      });
      
      const lineMaterial = new THREE.LineBasicMaterial({
        color: 0x4c6bff,
        transparent: true,
        opacity: 0.3,
      });
      
      // Create blockchain structure
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
        
        // Create connections between blocks
        if (i > 0 && blocks[i - 1].mesh) {
          const points = [];
          // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
          points.push(blocks[i - 1].mesh!.position);
          points.push(block.position);
          
          const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);
          const line = new THREE.Line(lineGeometry, lineMaterial);
          scene.add(line);
          
          connections.push({
            line: line,
            startBlock: blocks[i - 1],
            endBlock: blocks[i]
          });
        }
      }
      
      // Connect the last block to the first to complete the chain
      if (blocks.length > 0 && blocks[0].mesh && blocks[blocks.length - 1].mesh) {
        const points = [];
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        points.push(blocks[blocks.length - 1].mesh!.position);
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        points.push(blocks[0].mesh!.position);
        
        const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);
        const line = new THREE.Line(lineGeometry, lineMaterial);
        scene.add(line);
        
        connections.push({
          line: line,
          startBlock: blocks[blocks.length - 1],
          endBlock: blocks[0]
        });
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
      
      // Update connections
      connections.forEach(connection => {
        // Only update connections if both start and end blocks have meshes
        if (connection.startBlock.mesh && connection.endBlock.mesh) {
          const points = [];
          // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
          points.push(connection.startBlock.mesh!.position);
          // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
          points.push(connection.endBlock.mesh!.position);
          
          connection.line.geometry.setFromPoints(points);
          connection.line.geometry.attributes.position.needsUpdate = true;
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
    <>
      {/* 3D Background Canvas */}
      <div 
        ref={canvasRef} 
        className="fixed inset-0 pointer-events-none z-0"
        style={{ opacity: 0.4 }}
      />
      
      {/* Content */}
      <div className="relative z-10 p-8 pt-6 bg-white">
        <div className="max-w-7xl mx-auto">
          {/* Page Header */}
          <div className="flex flex-col gap-y-4 mb-10 items-center">
            <h1 className="text-4xl font-bold text-gray-800">
              Welcome to VAULT
            </h1>
            <p className="text-gray-500">
              Monitor and analyze crypto fund performance and compliance
            </p>
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
            {stats.map((stat, index) => (
              <Card 
                key={stat.label} 
                className={`p-6 shadow-md transition-all duration-300 ${
                  hoveredStat === index ? 'shadow-lg translate-y-[-5px]' : 'shadow'
                }`}
                onMouseEnter={() => setHoveredStat(index)}
                onMouseLeave={() => setHoveredStat(null)}
              >
                <div className="flex items-center gap-4">
                  <div className={`${stat.bgColor} p-3 rounded-lg`}>
                    <stat.icon className={`w-6 h-6 ${stat.color}`} />
                  </div>
                  <div>
                    <p className="text-gray-500 text-sm">{stat.label}</p>
                    <h3 className="text-2xl font-bold">{stat.value}</h3>
                  </div>
                </div>
              </Card>
            ))}
          </div>

          {/* Features Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => (
              <Link
                key={feature.label}
                href={feature.href}
              >
                <Card 
                  className={`p-6 hover:shadow-lg transition-all cursor-pointer bg-white ${
                    hoveredFeature === index ? 'shadow-md translate-y-[-5px]' : 'shadow-sm'
                  }`}
                  onMouseEnter={() => setHoveredFeature(index)}
                  onMouseLeave={() => setHoveredFeature(null)}
                >
                  <div className={`${feature.bgColor} w-12 h-12 rounded-lg flex items-center justify-center mb-4`}>
                    <feature.icon className={`w-6 h-6 ${feature.color}`} />
                  </div>
                  <h3 className="text-lg font-semibold mb-2">{feature.label}</h3>
                  <p className="text-gray-500 text-sm">{feature.description}</p>
                </Card>
              </Link>
            ))}
          </div>
        </div>
      </div>
    </>
  );
}