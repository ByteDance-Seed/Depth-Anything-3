#!/usr/bin/env python3
"""
Depth Anything 3 - ZeroMQ Backend Server
High-performance inference server using ZeroMQ for low-latency communication.
"""
import zmq
import numpy as np
import torch
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from depth_anything_3.api import DepthAnything3

import logging

logger = logging.getLogger(__name__)

class DepthAnythingZMQServer:
    def __init__(self, model_name="da3-large", device="cuda", port=5555):
        """
        Initialize ZeroMQ server for Depth Anything 3.
        
        Args:
            model_name: Model to load (da3-small, da3-base, da3-large, da3-giant)
            device: Device to use (cuda or cpu)
            port: ZeroMQ port to bind to
        """
        self.device = device
        self.port = port
        
        # Initialize ZeroMQ context and socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        
        # Load model
        self.model = DepthAnything3.from_pretrained(f"depth-anything/{model_name.upper()}")
        self.model = self.model.to(device=device)
        self.model.eval()
        
        logger.info(f"Server ready: model={model_name}, device={device}, port={port}")

    def log_vram_usage(self):
        """Log current VRAM usage."""
        if self.device == 'cuda':
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            logger.debug(f"VRAM: Allocated={allocated:.1f}MB, Reserved={reserved:.1f}MB")

    def process_request(self, request):
        """
        Process incoming request and return depth prediction.
        """
        try:
            command = request.get('command', '')
            
            if command == 'ping':
                return {'status': 'success', 'message': 'pong'}
            
            elif command == 'inference_batch':
                images = request['images'] # List of images
                process_res = request.get('process_res', 504)
                
                # Extract intrinsics and extrinsics
                intrinsics_list = request.get('intrinsics')
                extrinsics_list = request.get('extrinsics')
                
                intrinsics = None
                if intrinsics_list is not None:
                    intrinsics = np.stack(intrinsics_list).astype(np.float32)
                    
                extrinsics = None
                if extrinsics_list is not None:
                    extrinsics = np.stack(extrinsics_list).astype(np.float32)

                batch_size = len(images)
                
                logger.debug(f"Processing batch of {batch_size} images (res={process_res})")
                self.log_vram_usage()
                
                # Run inference
                with torch.no_grad():
                    prediction = self.model.inference(
                        images,
                        intrinsics=intrinsics,
                        extrinsics=extrinsics,
                        export_dir=None,
                        export_format="mini_npz",
                        process_res=process_res,
                        show_cameras=False
                    )
                
                # Extract results (list of depths and intrinsics)
                depths = [d.astype(np.float32) for d in prediction.depth]
                intrinsics = [i.astype(np.float32) for i in prediction.intrinsics]
                
                return {
                    'status': 'success',
                    'depth': depths,
                    'intrinsics': intrinsics
                }
            
            elif command == 'inference':
                image = request['image']
                process_res = request.get('process_res', 504)
                
                logger.debug(f"Processing single image (res={process_res})")
                self.log_vram_usage()
                
                # Run inference
                with torch.no_grad():
                    prediction = self.model.inference(
                        [image],
                        export_dir=None,
                        export_format="mini_npz",
                        process_res=process_res,
                        show_cameras=False
                    )
                
                # Extract results
                depth = prediction.depth[0]  # (H, W)
                intrinsics = prediction.intrinsics[0]  # (3, 3)
                
                return {
                    'status': 'success',
                    'depth': depth.astype(np.float32),
                    'intrinsics': intrinsics.astype(np.float32)
                }
            
            else:
                logger.warning(f"Unknown command received: {command}")
                return {
                    'status': 'error',
                    'error': f'Unknown command: {command}'
                }
        
        except Exception as e:
            import traceback
            error_msg = f"Error processing request: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"❌ {error_msg}")
            return {
                'status': 'error',
                'error': error_msg
            }
    
    def run(self):
        """Run the server loop."""
        request_count = 0
        try:
            while True:
                # Wait for request
                message = self.socket.recv_pyobj()
                request_count += 1
                
                # Process request
                response = self.process_request(message)
                
                # Send response
                self.socket.send_pyobj(response)
                
        except KeyboardInterrupt:
            logger.info(f"Server stopped by user (processed {request_count} requests)")
        except Exception as e:
            logger.error(f"Server error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        self.socket.close()
        self.context.term()
        logger.info("Server closed")

def main():
    parser = argparse.ArgumentParser(description='Depth Anything 3 ZeroMQ Server')
    parser.add_argument('--model', type=str, default='da3-large',
                        choices=['da3-small', 'da3-base', 'da3-large', 'da3-giant',
                                 'da3nested-giant-large'],
                        help='Model name to load')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for inference')
    parser.add_argument('--port', type=int, default=5555,
                        help='ZeroMQ port to bind to')
    parser.add_argument('--log-level', type=str, default='WARNING',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging verbosity level (default: WARNING)')
    
    args = parser.parse_args()
    
    # Configure logging based on argument
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Create and run server
    server = DepthAnythingZMQServer(
        model_name=args.model,
        device=args.device,
        port=args.port
    )
    server.run()

if __name__ == '__main__':
    main()
