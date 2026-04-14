# test_my_site.py
import asyncio
import os
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_my_site(site_number, site_name):
    current_env = os.environ.copy()
    server_params = StdioServerParameters(
        command="/global/common/software/nersc9/pytorch/2.8.0/bin/python3",
        args=["/global/homes/h/hvtran/RCSFA/mcp/usgs-water-mcp/main.py"],
        env=current_env
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            print(f"\n{'='*70}")
            print(f"USGS Site: {site_number} - {site_name}")
            print(f"{'='*70}\n")
            
            # Get current conditions
            result = await session.call_tool(
                "fetch_usgs_data",
                arguments={
                    "sites": site_number,
                    "parameter_codes": "00060,00065,00010",
                    "period": "P1D"
                }
            )
            
            data = json.loads(result.content[0].text)
            if 'value' in data and 'timeSeries' in data['value']:
                for ts in data['value']['timeSeries']:
                    var_name = ts['variable']['variableDescription']
                    unit = ts['variable'].get('unit', {}).get('unitCode', 'N/A')
                    values = ts['values'][0]['value']
                    
                    if values:
                        latest = values[-1]
                        print(f"{var_name}: {latest['value']} {unit}")
                        print(f"  Measured: {latest['dateTime']}\n")

# Test any site
asyncio.run(test_my_site("01646500", "Potomac River"))