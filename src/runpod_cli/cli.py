"""CLI for Sigil Experiments RunPod management."""

import re
import sys
from typing import Annotated, Optional

import runpod
import typer
from rich.console import Console
from rich.table import Table

from runpod_cli import __version__
from runpod_cli.config import get_settings


app = typer.Typer(
    name="pod",
    help="Sigil Experiments RunPod CLI - GPU pod management for training experiments",
    no_args_is_help=True,
)

console = Console()

POD_NAME_PREFIX = "sigil"


def init_runpod() -> None:
    """Initialize RunPod SDK with API key."""
    import logging

    # Suppress SDK debug output
    logging.getLogger("runpod").setLevel(logging.WARNING)

    settings = get_settings()

    if not settings.runpod_api_key:
        console.print("[red]Error: RUNPOD_API_KEY is not set[/red]")
        console.print("\nSet it in your .env file or as an environment variable:")
        console.print("  export RUNPOD_API_KEY=your_api_key_here")
        console.print("\nGet your API key from: https://www.runpod.io/console/user/settings")
        raise typer.Exit(1)

    runpod.api_key = settings.runpod_api_key


# =============================================================================
# Helper Functions
# =============================================================================

def get_templates() -> list[dict]:
    """Get templates using GraphQL API."""
    query = """
    query {
        myself {
            podTemplates {
                id
                name
                imageName
                isServerless
                volumeInGb
                containerDiskInGb
            }
        }
    }
    """
    response = runpod.api.graphql.run_graphql_query(query)
    return response.get("data", {}).get("myself", {}).get("podTemplates", [])


def get_gpu_types() -> list[dict]:
    """Fetch GPU types with pricing and availability via GraphQL."""
    query = """
    query GpuTypes {
        gpuTypes {
            id
            displayName
            memoryInGb
            communityCloud
            communitySpotPrice
            secureCloud
            securePrice
            secureSpotPrice
            lowestPrice(input: { gpuCount: 1 }) {
                stockStatus
            }
        }
    }
    """
    response = runpod.api.graphql.run_graphql_query(query)
    return response.get("data", {}).get("gpuTypes", [])


def get_datacenter_gpu_availability(datacenter_id: str) -> dict[str, str]:
    """Fetch GPU availability for a specific datacenter.

    Returns a dict mapping GPU type ID to stock status.
    """
    query = """
    query {
        dataCenters {
            id
            gpuAvailability {
                gpuTypeId
                stockStatus
            }
        }
    }
    """
    response = runpod.api.graphql.run_graphql_query(query)
    datacenters = response.get("data", {}).get("dataCenters", [])

    for dc in datacenters:
        if dc.get("id") == datacenter_id:
            availability = {}
            for gpu in dc.get("gpuAvailability", []):
                gpu_id = gpu.get("gpuTypeId")
                status = gpu.get("stockStatus")
                if gpu_id:
                    availability[gpu_id] = status
            return availability

    return {}


def resolve_template_id(identifier: str) -> tuple[str, str]:
    """Resolve a template identifier (ID or name) to (template_id, template_name)."""
    try:
        templates = get_templates()
    except Exception as e:
        console.print(f"[red]Error fetching templates: {e}[/red]")
        raise typer.Exit(1)

    # First try to match by ID
    for t in templates:
        if t.get("id") == identifier:
            return t.get("id"), t.get("name", "unnamed")

    # Then try to match by name
    for t in templates:
        if t.get("name") == identifier:
            return t.get("id"), t.get("name")

    return None, None


def resolve_pod_id(identifier: str) -> tuple[str, str]:
    """Resolve a pod identifier (ID or name) to (pod_id, pod_name)."""
    try:
        pods = runpod.get_pods()
    except Exception as e:
        console.print(f"[red]Error fetching pods: {e}[/red]")
        raise typer.Exit(1)

    # First try to match by ID
    for pod in pods:
        if pod.get("id") == identifier:
            return pod.get("id"), pod.get("name", "unnamed")

    # Then try to match by name
    for pod in pods:
        if pod.get("name") == identifier:
            return pod.get("id"), pod.get("name")

    # Try prefix match on name (e.g. "sigil" matches "sigil-phase1")
    for pod in pods:
        if pod.get("name", "").startswith(identifier):
            return pod.get("id"), pod.get("name")

    console.print(f"[red]Pod not found: {identifier}[/red]")
    console.print("[dim]Use 'pod list' to see available pods[/dim]")
    raise typer.Exit(1)


def load_env_file(path: str) -> dict[str, str]:
    """Load environment variables from a file.

    Supports KEY=VALUE format, one per line. Lines starting with # are comments.
    """
    env_vars = {}
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    env_vars[key.strip()] = value.strip()
    except FileNotFoundError:
        pass
    return env_vars


def get_sigil_pods() -> list[dict]:
    """Get all pods with the sigil prefix."""
    try:
        pods = runpod.get_pods()
    except Exception as e:
        console.print(f"[red]Error fetching pods: {e}[/red]")
        raise typer.Exit(1)

    return [
        p for p in pods
        if p.get("name", "").startswith(POD_NAME_PREFIX)
    ]


# =============================================================================
# List Commands
# =============================================================================

@app.command("list")
def list_pods():
    """List all pods with details."""
    init_runpod()

    console.print("[cyan]Fetching pods...[/cyan]")

    try:
        pods = runpod.get_pods()
    except Exception as e:
        console.print(f"[red]Error fetching pods: {e}[/red]")
        raise typer.Exit(1)

    if not pods:
        console.print("[yellow]No pods found.[/yellow]")
        console.print("\nTo create a pod, run:")
        console.print("  pod create <template>")
        return

    table = Table(title="RunPod Pods")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("ID", style="dim")
    table.add_column("Status", style="bold")
    table.add_column("GPU", style="green")
    table.add_column("vCPU", justify="right")
    table.add_column("RAM", justify="right")
    table.add_column("$/hr", justify="right", style="yellow")

    for pod in pods:
        name = pod.get("name", "unnamed")
        pod_id = pod.get("id", "?")

        status = pod.get("desiredStatus", pod.get("status", "?"))
        if status == "RUNNING":
            status_display = f"[green]{status}[/green]"
        elif status == "EXITED":
            status_display = f"[yellow]{status}[/yellow]"
        else:
            status_display = f"[dim]{status}[/dim]"

        gpu_type = pod.get("machine", {}).get("gpuDisplayName", "?") if pod.get("machine") else "?"
        gpu_count = pod.get("gpuCount", 1)
        gpu_display = f"{gpu_count}x {gpu_type}" if gpu_count > 1 else gpu_type

        vcpu_count = pod.get("vcpuCount", 0)
        memory_gb = pod.get("memoryInGb", 0)
        vcpu_display = str(vcpu_count) if vcpu_count else "-"
        ram_display = f"{memory_gb}GB" if memory_gb else "-"

        cost_per_hr = pod.get("costPerHr", 0)
        cost_display = f"${cost_per_hr:.2f}" if cost_per_hr else "-"

        table.add_row(name, pod_id, status_display, gpu_display, vcpu_display, ram_display, cost_display)

    console.print(table)
    console.print(f"\n[dim]Total pods: {len(pods)}[/dim]")


@app.command("list-templates")
def list_templates():
    """List available templates."""
    init_runpod()

    console.print("[cyan]Fetching templates...[/cyan]")

    try:
        templates = get_templates()
    except Exception as e:
        console.print(f"[red]Error fetching templates: {e}[/red]")
        raise typer.Exit(1)

    if not templates:
        console.print("[yellow]No templates found.[/yellow]")
        return

    table = Table(title="Available Templates")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("ID", style="dim")
    table.add_column("Image", style="dim", max_width=50)

    for template in templates:
        name = template.get("name", "unnamed")
        template_id = template.get("id", "?")
        image = template.get("imageName", "-")
        if len(image) > 50:
            image = image[:47] + "..."
        table.add_row(name, template_id, image)

    console.print(table)
    console.print(f"\n[dim]Total templates: {len(templates)}[/dim]")


@app.command("list-gpus")
def list_gpus(
    cloud_type: Annotated[Optional[str], typer.Option("--cloud", "-c", help="Filter by cloud type: SECURE or COMMUNITY")] = None,
    region: Annotated[Optional[str], typer.Option("--region", "-r", help="Filter by region/datacenter (e.g., US-TX-3)")] = None,
):
    """List available GPU types with pricing."""
    init_runpod()
    settings = get_settings()

    target_cloud = cloud_type or settings.default_cloud_type
    target_region = region

    console.print("[cyan]Fetching GPUs...[/cyan]")

    try:
        gpus = get_gpu_types()
    except Exception as e:
        console.print(f"[red]Error fetching GPUs: {e}[/red]")
        raise typer.Exit(1)

    # Get regional availability if region specified
    regional_availability: dict[str, str] = {}
    if target_region:
        try:
            regional_availability = get_datacenter_gpu_availability(target_region)
        except Exception:
            pass

    if not gpus:
        console.print("[yellow]No GPUs found.[/yellow]")
        return

    # Filter by cloud type
    filtered_gpus = []
    for gpu in gpus:
        if target_cloud == "SECURE" and gpu.get("secureCloud"):
            filtered_gpus.append(gpu)
        elif target_cloud == "COMMUNITY" and gpu.get("communityCloud"):
            filtered_gpus.append(gpu)
        elif not target_cloud:
            filtered_gpus.append(gpu)

    # If region specified, filter to only GPUs available there
    if target_region and regional_availability:
        filtered_gpus = [g for g in filtered_gpus if g.get("id") in regional_availability]

    title = f"Available GPUs ({target_cloud} Cloud)"
    if target_region:
        title += f" in {target_region}"

    table = Table(title=title, show_lines=False)
    table.add_column("GPU ID (use with --gpu)", style="cyan", no_wrap=True)
    table.add_column("VRAM", justify="right", style="green", no_wrap=True)
    table.add_column("$/hr", justify="right", style="yellow", no_wrap=True)
    table.add_column("Spot", justify="right", style="yellow", no_wrap=True)
    table.add_column("Stock", justify="center", no_wrap=True)

    for gpu in sorted(filtered_gpus, key=lambda x: x.get("id", "")):
        gpu_id = gpu.get("id", "?")
        memory_gb = gpu.get("memoryInGb", 0)
        memory_display = f"{memory_gb}GB" if memory_gb else "-"

        if target_cloud == "SECURE":
            on_demand = gpu.get("securePrice")
            spot = gpu.get("secureSpotPrice")
        else:
            on_demand = gpu.get("communitySpotPrice")
            spot = gpu.get("communitySpotPrice")

        on_demand_display = f"${on_demand:.2f}" if on_demand else "-"
        spot_display = f"${spot:.2f}" if spot else "-"

        if target_region and gpu_id in regional_availability:
            stock_status = regional_availability[gpu_id]
        else:
            lowest_price = gpu.get("lowestPrice", {}) or {}
            stock_status = lowest_price.get("stockStatus", "")

        if stock_status == "High":
            stock_display = "[green]High[/green]"
        elif stock_status == "Medium":
            stock_display = "[yellow]Med[/yellow]"
        elif stock_status == "Low":
            stock_display = "[red]Low[/red]"
        else:
            stock_display = "[dim]-[/dim]"

        table.add_row(gpu_id, memory_display, on_demand_display, spot_display, stock_display)

    console.print(table)
    console.print(f"\n[dim]Showing {len(filtered_gpus)} GPU types. Use the 'GPU ID' column value with --gpu[/dim]")
    if target_region:
        console.print(f"[dim]Stock status is for {target_region}. Use --region to check other datacenters.[/dim]")
    else:
        console.print(f"[dim]Prices are per hour. Use --region to filter by datacenter.[/dim]")


# =============================================================================
# Create Command
# =============================================================================

@app.command("create")
def create_pod(
    name: Annotated[str, typer.Argument(help="Pod name (prefixed with 'sigil-' automatically)")],
    template: Annotated[Optional[str], typer.Option("--template", "-t", help="Template ID or name")] = None,
    image: Annotated[Optional[str], typer.Option("--image", "-i", help="Docker image (default: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04)")] = None,
    gpu_type: Annotated[Optional[str], typer.Option("--gpu", "-g", help="GPU type")] = None,
    gpu_count: Annotated[Optional[int], typer.Option("--gpu-count", help="Number of GPUs")] = None,
    region: Annotated[Optional[str], typer.Option("--region", "-r", help="Data center region")] = None,
    container_disk: Annotated[Optional[int], typer.Option("--container-disk", help="Container disk size in GB")] = None,
    volume: Annotated[Optional[int], typer.Option("--volume", help="Volume size in GB")] = None,
    cloud_type: Annotated[Optional[str], typer.Option("--cloud-type", help="Cloud type: SECURE or COMMUNITY")] = None,
    env_file: Annotated[Optional[str], typer.Option("--env-file", "-e", help="Path to env file to inject into pod")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", "-d", help="Preview without creating")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation")] = False,
):
    """Create a new pod for experiments.

    NAME is used for the pod name (auto-prefixed with 'sigil-').

    Example:
        pod create phase1
        pod create phase1 --gpu "NVIDIA A100 80GB PCIe"
        pod create phase2 --template my-template --volume 300
        pod create phase1 --env-file .env --dry-run
    """
    init_runpod()
    settings = get_settings()

    # Resolve template or use Docker image directly
    default_image = "runpod/pytorch:2.8.0-py3.12-cuda12.8.1-devel-ubuntu22.04"
    docker_image = image or default_image
    template_id = None
    template_name = None

    if template:
        template_id, template_name = resolve_template_id(template)
        if template_id is None:
            console.print(f"[red]Template not found: {template}[/red]")
            console.print("[dim]Use 'pod list-templates' to see available templates[/dim]")
            raise typer.Exit(1)
    elif not image:
        # Try default template, fall back to Docker image
        template_id, template_name = resolve_template_id(settings.default_template)
        if template_id is None:
            console.print(f"[dim]Template '{settings.default_template}' not found, using Docker image[/dim]")

    # Apply defaults
    pod_name = f"{POD_NAME_PREFIX}-{name}"
    pod_gpu_type = gpu_type or settings.default_gpu_type
    pod_gpu_count = gpu_count or settings.default_gpu_count
    pod_region = region if region is not None else settings.default_region
    pod_container_disk = container_disk or settings.default_container_disk_gb
    pod_volume = volume or settings.default_volume_gb
    pod_cloud_type = cloud_type or settings.default_cloud_type

    # Load env file if specified
    pod_env = {}
    if env_file:
        pod_env = load_env_file(env_file)

    # Fetch GPU info for display
    gpu_info = None
    try:
        gpus = get_gpu_types()
        for g in gpus:
            if g.get("id") == pod_gpu_type:
                gpu_info = g
                break
    except Exception:
        pass

    if dry_run:
        console.print("[bold yellow]DRY RUN - Preview only, no pod will be created[/bold yellow]")
        console.print()

    console.print(f"[bold]{'Would create' if dry_run else 'Creating'} pod:[/bold]")
    console.print(f"  Name: [cyan]{pod_name}[/cyan]")
    if template_id:
        console.print(f"  Template: [cyan]{template_name}[/cyan] ({template_id})")
    else:
        console.print(f"  Image: [cyan]{docker_image}[/cyan]")
    console.print(f"  GPU: [green]{pod_gpu_count}x {pod_gpu_type}[/green]")

    if gpu_info:
        vram = gpu_info.get("memoryInGb", 0)
        if vram:
            console.print(f"  VRAM: [green]{vram} GB[/green]")

        if pod_cloud_type == "SECURE":
            on_demand = gpu_info.get("securePrice")
            spot = gpu_info.get("secureSpotPrice")
        else:
            on_demand = gpu_info.get("communitySpotPrice")
            spot = gpu_info.get("communitySpotPrice")

        # Get regional stock status
        regional_availability: dict[str, str] = {}
        if pod_region:
            try:
                regional_availability = get_datacenter_gpu_availability(pod_region)
            except Exception:
                pass

        if pod_region and pod_gpu_type in regional_availability:
            stock_status = regional_availability[pod_gpu_type]
        else:
            lowest_price = gpu_info.get("lowestPrice", {}) or {}
            stock_status = lowest_price.get("stockStatus", "")

        if stock_status == "High":
            console.print(f"  Stock: [green]High in {pod_region} - good availability[/green]")
        elif stock_status == "Medium":
            console.print(f"  Stock: [yellow]Medium in {pod_region} - may take time[/yellow]")
        elif stock_status == "Low":
            console.print(f"  Stock: [red]Low in {pod_region} - limited availability[/red]")

        if on_demand:
            per_pod = on_demand * pod_gpu_count
            console.print(f"  Pricing: [yellow]${per_pod:.2f}/hr[/yellow]")
            if spot:
                console.print(f"  Spot: [dim]${spot * pod_gpu_count:.2f}/hr[/dim]")

    console.print(f"  Region: [dim]{pod_region}[/dim]")
    console.print(f"  Cloud type: [dim]{pod_cloud_type}[/dim]")
    console.print(f"  Container disk: [dim]{pod_container_disk} GB[/dim]")
    console.print(f"  Volume: [dim]{pod_volume} GB[/dim]")

    if pod_env:
        console.print(f"  Env vars: [green]{len(pod_env)} from {env_file}[/green]")

    console.print()

    if dry_run:
        if pod_env:
            console.print("[bold]Environment variables:[/bold]")
            for key, value in pod_env.items():
                display_val = value if len(value) <= 50 else value[:47] + "..."
                console.print(f"  {key}={display_val}")
            console.print()
        console.print("[dim]Remove --dry-run to create the pod[/dim]")
        return

    if not yes:
        confirm = typer.confirm("Create pod?")
        if not confirm:
            console.print("[yellow]Cancelled[/yellow]")
            raise typer.Exit(0)

    try:
        create_kwargs = dict(
            name=pod_name,
            gpu_type_id=pod_gpu_type,
            gpu_count=pod_gpu_count,
            cloud_type=pod_cloud_type,
            container_disk_in_gb=pod_container_disk,
            volume_in_gb=pod_volume,
            env=pod_env if pod_env else None,
        )
        if pod_region:
            create_kwargs["data_center_id"] = pod_region
        if template_id:
            create_kwargs["template_id"] = template_id
        else:
            create_kwargs["image_name"] = docker_image
            create_kwargs["ports"] = "22/tcp"

        pod = runpod.create_pod(**create_kwargs)

        if pod:
            pod_id = pod.get("id", "?")
            console.print(f"\n[green]Pod created successfully![/green]")
            console.print(f"  ID: [cyan]{pod_id}[/cyan]")
            console.print(f"  Name: [cyan]{pod_name}[/cyan]")
            console.print()
            console.print(f"[dim]https://console.runpod.io/pods?id={pod_id}[/dim]")
        else:
            console.print("[red]Pod creation returned empty response[/red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error creating pod: {e}[/red]")
        console.print("\n[dim]Common issues:[/dim]")
        console.print("  - GPU type not available in region")
        console.print("  - Template ID is invalid")
        console.print("  - Insufficient account balance")
        raise typer.Exit(1)


# =============================================================================
# Stop/Start/Delete Commands
# =============================================================================

@app.command("stop")
def stop_pod(
    pod_identifier: Annotated[str, typer.Argument(help="Pod ID or name")],
):
    """Stop a running pod (preserves data, still charges for storage)."""
    init_runpod()

    pod_id, pod_name = resolve_pod_id(pod_identifier)

    console.print(f"[yellow]Stopping pod: {pod_name} ({pod_id})[/yellow]")

    try:
        runpod.stop_pod(pod_id)
        console.print(f"[green]Pod {pod_name} stopped[/green]")
        console.print("[dim]Note: Still charged for storage. Use 'rm' to fully remove.[/dim]")
    except Exception as e:
        console.print(f"[red]Error stopping pod: {e}[/red]")
        raise typer.Exit(1)


@app.command("start")
def start_pod(
    pod_identifier: Annotated[str, typer.Argument(help="Pod ID or name")],
):
    """Start/resume a stopped pod."""
    init_runpod()

    pod_id, pod_name = resolve_pod_id(pod_identifier)

    console.print(f"[cyan]Starting pod: {pod_name} ({pod_id})[/cyan]")

    try:
        runpod.resume_pod(pod_id)
        console.print(f"[green]Pod {pod_name} started[/green]")
    except Exception as e:
        console.print(f"[red]Error starting pod: {e}[/red]")
        raise typer.Exit(1)


@app.command("restart")
def restart_pod(
    pod_identifier: Annotated[str, typer.Argument(help="Pod ID or name")],
):
    """Restart a pod (stop then start)."""
    init_runpod()

    pod_id, pod_name = resolve_pod_id(pod_identifier)

    console.print(f"[yellow]Stopping pod: {pod_name} ({pod_id})[/yellow]")
    try:
        runpod.stop_pod(pod_id)
        console.print(f"[dim]Stopped[/dim]")
    except Exception as e:
        console.print(f"[dim]Stop skipped: {e}[/dim]")

    console.print(f"[cyan]Starting pod: {pod_name}[/cyan]")
    try:
        runpod.resume_pod(pod_id)
        console.print(f"[green]Pod {pod_name} restarted[/green]")
    except Exception as e:
        console.print(f"[red]Error starting pod: {e}[/red]")
        raise typer.Exit(1)


@app.command("rm")
def rm_pod(
    pod_identifier: Annotated[str, typer.Argument(help="Pod ID or name")],
    force: Annotated[bool, typer.Option("--force", "-f", help="Skip confirmation")] = False,
):
    """Stop and delete a pod permanently.

    All data outside the volume will be lost.
    """
    init_runpod()

    pod_id, pod_name = resolve_pod_id(pod_identifier)

    if not force:
        console.print(f"[bold red]Warning:[/bold red] This will permanently delete pod [cyan]{pod_name}[/cyan] ({pod_id})")
        console.print("All data outside network volumes will be lost.")
        confirm = typer.confirm("Are you sure?")
        if not confirm:
            console.print("[yellow]Cancelled[/yellow]")
            raise typer.Exit(0)

    console.print(f"[yellow]Stopping pod: {pod_name}[/yellow]")
    try:
        runpod.stop_pod(pod_id)
        console.print(f"[dim]Stopped[/dim]")
    except Exception:
        console.print(f"[dim]Stop skipped (may already be stopped)[/dim]")

    console.print(f"[red]Deleting pod: {pod_name}[/red]")
    try:
        runpod.terminate_pod(pod_id)
        console.print(f"[green]Pod {pod_name} removed[/green]")
    except Exception as e:
        console.print(f"[red]Error deleting pod: {e}[/red]")
        raise typer.Exit(1)


@app.command("rm-all")
def rm_all(
    force: Annotated[bool, typer.Option("--force", "-f", help="Skip confirmation")] = False,
):
    """Remove all sigil-* pods.

    Finds all pods with names starting with 'sigil-' and terminates them.
    """
    init_runpod()

    sigil_pods = get_sigil_pods()

    if not sigil_pods:
        console.print("[yellow]No sigil pods found.[/yellow]")
        return

    console.print(f"[bold]Found {len(sigil_pods)} sigil pod(s) to remove:[/bold]")
    for p in sigil_pods:
        console.print(f"  {p.get('name', '?')} ({p.get('id', '?')})")
    console.print()

    if not force:
        confirm = typer.confirm(f"Terminate {len(sigil_pods)} pod(s)?")
        if not confirm:
            console.print("[yellow]Cancelled[/yellow]")
            raise typer.Exit(0)

    success_count = 0
    fail_count = 0

    for pod in sigil_pods:
        pod_id = pod.get("id")
        pod_name = pod.get("name", "?")
        try:
            console.print(f"[red]Terminating {pod_name}...[/red]")
            runpod.terminate_pod(pod_id)
            console.print(f"  [green]Done[/green]")
            success_count += 1
        except Exception as e:
            console.print(f"  [red]Failed: {e}[/red]")
            fail_count += 1

    console.print()
    if fail_count == 0:
        console.print(f"[bold green]Removed {success_count} pod(s)[/bold green]")
    else:
        console.print(f"[yellow]Removed {success_count} pod(s), {fail_count} failed[/yellow]")


# =============================================================================
# Info Command
# =============================================================================

@app.command("info")
def pod_info(
    pod_identifier: Annotated[str, typer.Argument(help="Pod ID or name")],
):
    """Show detailed information about a pod."""
    init_runpod()

    pod_id, _ = resolve_pod_id(pod_identifier)

    console.print(f"[cyan]Fetching pod info...[/cyan]")

    try:
        pod = runpod.get_pod(pod_id)
    except Exception as e:
        console.print(f"[red]Error fetching pod: {e}[/red]")
        raise typer.Exit(1)

    if not pod:
        console.print(f"[red]Pod not found: {pod_id}[/red]")
        raise typer.Exit(1)

    console.print()
    console.print(f"[bold]{pod.get('name', 'unnamed')}[/bold]")
    console.print(f"  ID: {pod.get('id', '?')}")
    console.print(f"  Status: {pod.get('desiredStatus', pod.get('status', '?'))}")

    # GPU info
    machine = pod.get("machine", {}) or {}
    console.print(f"  GPU: {pod.get('gpuCount', 1)}x {machine.get('gpuDisplayName', '?')}")

    # CPU/RAM info
    vcpu = pod.get("vcpuCount", 0)
    memory = pod.get("memoryInGb", 0)
    if vcpu or memory:
        console.print(f"  CPU: {vcpu} vCPU")
        console.print(f"  RAM: {memory} GB")

    # Cost
    cost_per_hr = pod.get("costPerHr", 0)
    if cost_per_hr:
        console.print(f"  Cost: ${cost_per_hr:.3f}/hr")

    # Template
    template_id = pod.get("templateId")
    if template_id:
        console.print(f"  Template: {template_id}")

    # Image
    image_name = pod.get("imageName")
    if image_name:
        console.print(f"  Image: {image_name}")

    # Runtime info
    runtime = pod.get("runtime", {}) or {}
    if runtime:
        uptime = runtime.get("uptimeInSeconds", 0)
        if uptime:
            hours = uptime // 3600
            minutes = (uptime % 3600) // 60
            console.print(f"  Uptime: {hours}h {minutes}m")

        ports = runtime.get("ports", [])
        if ports:
            console.print("  Ports:")
            for port in ports:
                port_num = port.get("privatePort") or port.get("port")
                ip = port.get("ip")
                public_port = port.get("publicPort")
                if ip and public_port:
                    console.print(f"    {port_num} -> {ip}:{public_port}")

    console.print()
    console.print(f"[dim]https://console.runpod.io/pods?id={pod_id}[/dim]")


# =============================================================================
# SSH Command
# =============================================================================

@app.command("ssh")
def ssh_pod(
    pod_identifier: Annotated[str, typer.Argument(help="Pod ID or name")],
):
    """Print SSH command for connecting to a pod.

    Example:
        pod ssh phase1
        eval $(pod ssh phase1)  # connect directly
    """
    import json

    init_runpod()

    pod_id, pod_name = resolve_pod_id(pod_identifier)

    try:
        pod = runpod.get_pod(pod_id)
    except Exception as e:
        console.print(f"[red]Error fetching pod: {e}[/red]")
        raise typer.Exit(1)

    if not pod:
        console.print(f"[red]Pod not found[/red]")
        raise typer.Exit(1)

    runtime = pod.get("runtime", {}) or {}
    ports = runtime.get("ports", []) or []

    # Find SSH port (22)
    ssh_port = None
    ssh_ip = None
    for port in ports:
        if port.get("privatePort") == 22 or port.get("port") == 22:
            ssh_ip = port.get("ip")
            ssh_port = port.get("publicPort")
            break

    if not ssh_ip or not ssh_port:
        console.print(f"[yellow]No SSH port found for {pod_name}.[/yellow]")
        console.print("[dim]Pod may still be starting, or SSH is not exposed.[/dim]")
        raise typer.Exit(1)

    ssh_cmd = f"ssh root@{ssh_ip} -p {ssh_port} -o StrictHostKeyChecking=no"
    console.print(f"[green]{ssh_cmd}[/green]")
    console.print(f"\n[dim]Or run: eval $(pod ssh {pod_identifier})[/dim]")


# =============================================================================
# Version Command
# =============================================================================

@app.command("version")
def version():
    """Show version information."""
    console.print(f"Sigil Experiments RunPod CLI v{__version__}")

    settings = get_settings()
    console.print(f"Default region: {settings.default_region}")
    console.print(f"Default GPU: {settings.default_gpu_type}")
    console.print(f"Default volume: {settings.default_volume_gb} GB")

    if settings.runpod_api_key:
        key = settings.runpod_api_key
        masked = key[:4] + "..." + key[-4:] if len(key) > 8 else "***"
        console.print(f"API Key: {masked}")
    else:
        console.print("[yellow]API Key: Not set[/yellow]")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
